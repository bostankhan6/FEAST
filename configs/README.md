# configs/

Architecture search-space definitions for the OFA-style supernet
(`GenericOFAResNet`, `src/feast/elastic_nn/generic_ofa_network.py`).

**These JSON files are not read by `train.py`.** Training's architecture
parameters come entirely from CLI flags in each `experiments/*.sh` script
(`--supernet_num_stages`, `--supernet_width_multiplier_choices`, etc. — see
`train.py:create_model()`), which happen to encode the same values as
`configs/supernets/4-stage-supernet-cifar100-v2.json`. These JSON files are
consumed only by auxiliary scripts: subnet-cache generation
(`scripts/cache_generation/`), the standalone architecture searcher
(`src/feast/nas/search_single_subnet.py`), cross-method footprint analysis
(`scripts/client_param_footprint.py`), the sub-supernet reproduction script
(`scripts/reproduce_sub_supernet_communication_reduction.py`), and
`tests/test_sub_supernet.py`.

## Subfolders

- **`supernets/`** — the two supernet config JSON files. See its own
  `README.md` for the field-by-field breakdown and how the two files relate.
