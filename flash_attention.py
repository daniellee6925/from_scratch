import torch

import triton
import triton.language as t1


@triton.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    M,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: t1.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    t1.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)
    
    # this indicates which block in the sequence length to process
    block_index_q = t1.program_id(0)

    # this indicates which head and batch to process. Each program is associated with a single head of a single batch
    index_batch_head = t1.program_id(1)
    
    # this indicates which batch this program is associated with (each batch has NUM_HEAD heads)
    index_batch = index_batch_head // NUM_HEADS
    

class TritionAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, casual, softmax_scale):
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]
        
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        
        O = torch.empty_like(Q)
        stage = 3 if casual else 1
        
        # how many programs can work in parallel 
        grid = lambda args:(
            # ceil(SEQ_LEN/BLOCK_SIZE_Q) how many blocks of Q we have 
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), #which group of queires 
            BATCH_SIZE * NUM_HEADS, #which head of which batch element 
            1, # Z in the CUDA launch grid
        )
        # Number of Parallel Programs (BATCH_SIZE * NUM_HEADS * NUM_BLOCKS_Q)
        
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device = Q.device, dtype = torch.float32
        )
        
        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale = softmax_scale,
            M=M,
            O=O,
            stride_Q_batch = Q.stride(0),
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )
        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.casual = casual
        return O

def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, casual, dtype = torch.float16):
    Q =(
        torch.empty(
            BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype = dtype, device ='cuda'
        ).normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K =(
        torch.empty(
            BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype = dtype, device ='cuda'
        ).normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V =(
        torch.empty(
            BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype = dtype, device ='cuda'
        ).normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    
    softmax_scale = 1 / (HEAD_DIM**0.5) #QK^T/sqrt(head_dim)
    d0 = torch.rand_like(Q)
    
    #naive implementation
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device='cuda'))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    if casual:
        P[:, :, MASK == 0] = float('inf')
    P = torch.softmax(P.float(), dim = -1).half()
    ref_0 = torch.matmul(P, V)
    ref_0.backward(d0) # compute gradients 
    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None
    
    # trition implementation
    tri_out = TritionAttention.apply(Q, K, V, casual, softmax_scale).half()
    tri_out.backward(d0)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None
    
    #compare 
    rtol = 0.0
    atol = 1e-2
    
    assert torch.allclose(ref_0, tri_out, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dV, tri_dV, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dK, tri_dK, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol=atol, rtol=rtol)