from __future__ import annotations
import io,struct
import numpy as np
import torch
from .sparse_sync import SparsePayload,SparseTensor
def serialize_payload(payload: SparsePayload)->bytes:
    buffer=io.BytesIO(); buffer.write(b"FSS1"); buffer.write(struct.pack("<H",len(payload.template_shapes)))
    for name,shape in sorted(payload.template_shapes.items()):
        encoded=name.encode("utf-8"); buffer.write(struct.pack("<H",len(encoded))); buffer.write(encoded); buffer.write(struct.pack("<B",len(shape))); buffer.write(struct.pack(f"<{len(shape)}I",*shape))
    buffer.write(struct.pack("<H",len(payload.tensors)))
    for name,item in sorted(payload.tensors.items()):
        encoded=name.encode("utf-8"); values=item.values.detach().cpu().contiguous().numpy(); indices=item.indices.detach().cpu().to(torch.int32).contiguous().numpy(); dtype=str(values.dtype).encode("ascii")
        buffer.write(struct.pack("<H",len(encoded))); buffer.write(encoded); buffer.write(struct.pack("<B",len(item.shape))); buffer.write(struct.pack(f"<{len(item.shape)}I",*item.shape)); buffer.write(struct.pack("<B",len(dtype))); buffer.write(dtype); buffer.write(struct.pack("<I",len(indices))); buffer.write(indices.tobytes()); buffer.write(values.tobytes())
    return buffer.getvalue()

def deserialize_payload(data: bytes)->SparsePayload:
    buffer=io.BytesIO(data)
    if buffer.read(4)!=b"FSS1": raise ValueError("invalid payload magic")
    templates={}
    for _ in range(struct.unpack("<H",buffer.read(2))[0]):
        name=buffer.read(struct.unpack("<H",buffer.read(2))[0]).decode("utf-8"); ndim=struct.unpack("<B",buffer.read(1))[0]; templates[name]=struct.unpack(f"<{ndim}I",buffer.read(4*ndim))
    tensors={}
    for _ in range(struct.unpack("<H",buffer.read(2))[0]):
        name=buffer.read(struct.unpack("<H",buffer.read(2))[0]).decode("utf-8"); ndim=struct.unpack("<B",buffer.read(1))[0]; shape=struct.unpack(f"<{ndim}I",buffer.read(4*ndim)); dtype=np.dtype(buffer.read(struct.unpack("<B",buffer.read(1))[0]).decode("ascii")); count=struct.unpack("<I",buffer.read(4))[0]; indices=torch.from_numpy(np.frombuffer(buffer.read(count*4),dtype=np.int32).copy()); values=torch.from_numpy(np.frombuffer(buffer.read(count*dtype.itemsize),dtype=dtype).copy()); tensors[name]=SparseTensor(shape,indices,values)
    if buffer.read(1): raise ValueError("trailing payload bytes")
    return SparsePayload(tensors,templates)
def payload_bytes(payload: SparsePayload)->int: return len(serialize_payload(payload))


def serialize_dense_update(tensors: dict[str,torch.Tensor])->bytes:
    # Dense wire format with explicit names, shapes, dtypes and values.
    buffer=io.BytesIO(); buffer.write(b"FSD1"); buffer.write(struct.pack("<H",len(tensors)))
    for name,tensor in sorted(tensors.items()):
        encoded=name.encode("utf-8")
        values=tensor.detach().cpu().contiguous().numpy()
        dtype=str(values.dtype).encode("ascii")
        shape=tuple(int(x) for x in values.shape)
        buffer.write(struct.pack("<H",len(encoded))); buffer.write(encoded)
        buffer.write(struct.pack("<B",len(shape))); buffer.write(struct.pack(f"<{len(shape)}I",*shape))
        buffer.write(struct.pack("<B",len(dtype))); buffer.write(dtype)
        raw=values.tobytes()
        buffer.write(struct.pack("<Q",len(raw))); buffer.write(raw)
    return buffer.getvalue()

def dense_update_bytes(tensors: dict[str,torch.Tensor])->int:
    return len(serialize_dense_update(tensors))
