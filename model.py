import torch
import torch.nn as nn
import torch.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
	dim: int = 4096
	n_layers: int = 32
	n_heads: int = 32 # Number of heads for the queries
	n_kv_heads: Optional[int] = None # Number of heads for the K and V
	vocab_size: int = -1 # This will be set when we load the tokenizer
	multiple_of: int = 256
	ffn_dim_multiplier: Optional[float] = None
	norm_eps: float = 1e-5

	# Needed for KV cache
	max_batch_size: int = 32
	max_seq_len: int = 2048

	device: str = None

def precomputer_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
	# As written in the paper, the dimension of the embedding must be even
	assert head_dim % 2 == 0, "Dimension must be divisible by 2"
	# Build the theta parameters
	# Formula: theta_i = [10000^ (-2(i-1)/dim) for i = [1, 2, ..., dim/2]
	# Shape: (Head dim / 2)
	# theta = [theta1, theta2, ..., thetad/2]
	theta_numerator = torch.arange(0, head_dim, 2).float()
	theta = 1.0 / (theta ** (theta_numerator/head_dim)).to(device)

	# Construct the positions ("m" parameter)
	# Shape: (seq_len)
	m = torch.arange(seq_len, device=device) # m = [1, 2, ..., seq_len]
	# Multiply each theta by each position using the outer product to get all the combinations of m and theta.
	# Shape: (seq_len) outer_product * (head_dim/2) --> (seq_len, head_dim/2)
	freqs = torch.outer(m, theta).float()
	# We can compute complex numbers in the polar coordinate c = R * exp(i * m * theta), where R = 1 as follows:
	# (seq_len, head_dim/2) -> (seq_len, head_dim/2)
	freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
	return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
	# Transformation 1
	# (B, seq_len, H, head_dim) --> (B, seq_len, H, head_dim/2)
	x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
	# (B, seq_len, head_dim/2) --> (1, seq_len, 1, head_dim/2)
	freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
	# (B, seq_len, H, head_dim/2) --> (1, seq_len, 1, head_dim/2) == (B, seq_len, H, head_dim/2)
	x_rotated = x_complex * freqs_complex
	# (B, seq_len, H, head_dim/2) -> (B, seq_len, H, head_dim/2, 2)
	x_out = torch.view_as_real(x_rotated)
	# (B, seq_len, H, head_dim/2, 2) --> (B, seq_len, H, head_dim)
	x_out = x_out.reshape(*x.shape)
	return x_out.type_as(x).to(device)

class RMSNorm(nn.Module):
	def __init__(self, dim: int, eps: float = 1e-6):
		super().__init__()
		# eps to avoid division by 0
		self.eps = eps
		# Gamma parameter
		self.weight = nn.Parameter(torch.ones(dim))

	def _norm(self, x: torch.Tensor):
		# (B, seq_len, dim)
		# rsqrt: 1 / sqrt(x)
		return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

	def forward(self, x: torch.Tensor):
		# (dim) * (B, seq_len, dim) = (B, seq_len, dim)
		return self.weight * self._norm(x.float()).type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
	batch_size, seq_len, n_kv_heads, head_dim = x.shape
	if n_rep == 1:
		return x
	else:
		return (
				# (B, seq_len, n_KV_heads, 1, head_dim)
				x[:, :, :, None, :].expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim).reshape(batch_size, seq_len, n_kv_heads*n_rep, head_dim)
			)



class SelfAttention(nn.Module):
	# This code will only work for inference since we are using KV cahce
	def __init__(self, args: ModelArgs):
		super().__init__()

		# indicate the number of heads for the key and value
		self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

		# Indicates the number of heads for the queries
		self.n_heads_q = args.n_heads

		# Indicates how many times the heads of keys and values should be repeated to match the head of the queries
		self.n_rep = self.n_heads_q // self.n_kv_heads

		# Indicates the dimension of each head
		self.head_dim = args.dim // args.n_heads

		self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
		self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
		self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
		self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

		self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
		self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

	def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
		batch_size, seq_len, _ = x.shape # (B, 1, dim) # seq_len is 1 in this case

		# Apply the wq, ek, and wv matrices to queries, keys, and values
		# (B, 1, dim) --> (B, 1, H_Q * Head_DIM)
		xq = self.wq(x)

		# (B, 1, dim) --> (B, 1, H_KV * Head_DIM)
		xk = self.wk(x)
		xv = self.wv(x)

		# (B, 1, HQ * head_dim) --> (B, 1, HQ, head_dim)
		xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
		xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

		# (B, 1, H_KV * head_dim) --> (B, 1, H_KV, head_dim)
		xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

		# Does not change the size of the tensors (cf. apply_rotary_embeddings)
		xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
		xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

		# Update the kv cache and replace the values for this token
		self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
		self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

		# Retrieve all the cached keys and values from start to this token for the computing attention up to this token
		# (Batch, start_pos+seq_len, H_KV, head_dim)
		keys = self.cache_k[:batch_size, 0:start_pos + seq_len]
		values = self.cache_v[:batch_size, 0:start_pos + seq_len]

		# Repeat the heads of the K and V to reach the number of the queries (to make it multi-headed attention)
		# This is not optimized way but we don't have any way to validate another method since we can not train our model.
		# Since it is difficult to validate group multi-query attetnion as only LLaMA's biggest model with 70B parameters support this.
		# Meta has also done the same way
		keys = repeat_kv(keys, self.n_rep)
		values = repeat_kv(values, self.n_rep)

		# (B, 1, H_Q, head_dim) --> (B, H_Q, 1, head_dim)
		xq = xq.transpose(1, 2)
		keys = keys.transpose(1, 2)
		values = values.transpose(1, 2)

		# (B, H_Q, 1, head_dim) @ (B, H_Q, head_dim, seq_len_KV) --> (B, H_Q, 1, seq_len_KV)
		scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
		scores = nn.functional.softmax(scores.float(), dim=-1).type_as(xq)

		# (B, H_Q, 1, seq_len) @ (B, H_Q, seq_len_KV, head_dim) --> (B, H_Q, 1, head_dim)
		output = torch.matmul(scores, values)

		output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
		return self.wo(output) # (B, 1, dim) -> (B, 1, dim)

class FeedForward(nn.Module):
	def __init__(self, args: ModelArgs):
		super().__init__()

		hidden_dim = 4 * args.dim # 4 times the dimension
		hidden_dim = int(2 * hidden_dim / 3) # then two-third of it
		if args.ffn_dim_multiplier is not None:
			hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

		# Round the hidden_dim to the nearest multiple of the multiple of parameter
		hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1)//args.multiple_of)

		self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
		self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
		self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

	def forward(self, x: torch.Tensor):
		swish = nn.functional.silu(self.w1(x))
		x_V = self.w3(x)
		x = swish * x_V
		x = self.w2(x)
		return x



class EncoderBlock(nn.Module):
	def __init__(self, args: ModelArgs):
		super().__init__()
		self.n_heads = args.n_heads
		self.dim = args.dim
		self.head_dim = args.dim // args.n_heads

		self.attention = SelfAttention(args)
		self.feed_forward = FeedForward(args)

		# Normalize BEFORE the self attention
		self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
		# Normalize BEFORE the feed forward block
		self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

	def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
		# (B, seq_len, dim) + (B, seq_len, dim) --> (B, seq_len, dim)
		h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
		out = h + self.feed_forward.forward(self.ffn_norm(h))
		return out

class Transformer(nn.Module):
	def __init__(self, args: ModelArgs) -> None:
		super().__init__()
		assert args.vocab_size != -1, "Vocab size must be set"

		self.args = args
		self.vocab_size = args.vocab_size
		self.n_layers = args.n_layers
		self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

		self.layers = nn.ModuleList()
		for _ in range(args.n_layers):
			self.layers.append(EncoderBlock(args))

		self.norm = RMSNorm(args.dim, eps=args.norm_eps) # eps to make sure we never divide by zero
		self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

		self.freqs_complex = precomputer_theta_pos_frequencies(self.args.dim//self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

	def forward(self, tokens: torch.Tensor, start_pos: int):
		# (B, seq_len)
		batch_size, seq_len = tokens.shape
		assert seq_len == 1, "Only one token at a time can be processed"

		# (B, seq_len) -> (B, seq_len, dim)
		h = self.tok_embeddings(tokens)

		# Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
		freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]

		# Consequently apply all the encoder layers
		for layer in self.layers:
			h = layer(h, start_pos, freqs_complex)
		h = self.norm(h)
		output = self.output(h).float()
		return output


 







