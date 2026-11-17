"""Optional actual PyTorch quantized-tensor export; no model download or training.

Requires an independently installed CPU-capable torch. RoundoffLab itself does
not depend on torch, and never imports torch while reading your matrices.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from roundofflab import study, render_report
from roundofflab.io import npy_bytes, write_artifacts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    import torch
    w=torch.tensor([[-256.,-250.]],dtype=torch.float32)
    tensor=torch.quantize_per_tensor(w,scale=2.,zero_point=128,dtype=torch.quint8)
    # Real dequantization includes the zero point; int_repr is not a weight matrix.
    correct=tensor.dequantize().detach().cpu().numpy()
    wrong=tensor.int_repr().double().cpu().numpy()*tensor.q_scale()
    base=w.numpy()
    discovery=np.array([[1.,0.]],dtype=np.float64)
    validation=np.array([[0.,1.]],dtype=np.float64)
    candidates={'dequantized':correct,'omitted-zero-point':wrong}
    result=study(base,discovery,validation,candidates=candidates,
                 title='An integer code is not a dequantized weight')
    for name,view in result['cohorts'].items():
        view['sample_label']='synthetic PyTorch CPU export / '+name
    result['warnings'].insert(0,'A tiny synthetic PyTorch integration test, not a pretrained model or deployed quantized kernel benchmark.')
    info={'torch_version':torch.__version__,'device':'cpu','integer_codes':tensor.int_repr().tolist(),
          'dequantized':correct.tolist(),'scale':tensor.q_scale(),'zero_point':tensor.q_zero_point()}
    payloads={'weights.npy':npy_bytes(base),'discovery.npy':npy_bytes(discovery),'validation.npy':npy_bytes(validation),
              'report.json':(json.dumps(result,indent=2,allow_nan=False)+'\n').encode(),
              'index.html':render_report(result).encode('utf-8'),'integration.json':json.dumps(info,indent=2).encode(),
              'candidates.json':json.dumps({k:k+'.npy' for k in candidates}).encode()}
    payloads.update({k+'.npy':npy_bytes(v) for k,v in candidates.items()})
    write_artifacts(args.out,payloads,force=args.force)
    print(json.dumps(info))
    print(f"Report: {args.out / 'index.html'}")


if __name__=='__main__':main()
