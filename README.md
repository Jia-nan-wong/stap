# STAP: Leveraging State-Transition Adversarial Perturbations for Asymmetric Website Fingerprinting Defenses (TNSM'25)

## Introduction

STAP is a state-transition adversarial perturbation framework designed to defend against website fingerprinting (WF) attacks. It leverages asymmetric perturbation strategies to balance defense effectiveness and overhead.

## Example
```
python pri-eval.py -d conn -gram 1 -dm ag -dataseed 0
```

## Citation
If you find this repo useful, please cite our paper via
```bibtex
@article{huang2025stap,
  title={STAP: Leveraging State-Transition Adversarial Perturbations for Asymmetric Website Fingerprinting Defenses},
  author={Huang, Jianan and Liu, Weiwei and Liu, Guangjie and Gao, Bo and Nie, Fengyuan and Mellia, Marco},
  journal={IEEE Transactions on Network and Service Management},
  year={2025},
  note={Early Access},
  doi={10.1109/TNSM.2025.3597075}
}
```
## Acknowledgement

We appreciate the following papers for their valuable code base:

- J.-P. Smith, L. Dolfi, P. Mittal, and A. Perrig. *Qcsd: A quic client-side website-fingerprinting defence framework.* In Proc. USENIX Security Symp., 2022, pp. 771–789.
- S. Oh, M. Lee, H. Lee, E. Bertino, and H. Kim. *Appsniffer: Towards robust mobile app fingerprinting against vpn.* In Proc. ACM Web Conf. (WWW), 2023, pp. 2318–2328.
- X. Yun, Y. Wang, Y. Zhang, C. Zhao, and Z. Zhao. *Encrypted tls traffic classification on cloud platforms.* IEEE/ACM Trans. Net., vol. 31, pp. 164–177, 2022.

## Contact
If you have any questions, please feel free to open an issue. We will respond as soon as possible.
