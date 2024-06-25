- Fetch label cube from hcp new
- Apply label map provided

```
def DK2synth(label, device):
    max_value = 103

    # Initialize the idx tensor with 1s
    idx = torch.ones(max_value + 1, dtype=torch.long).to(device)

    # Now set the other mappings
    idx[0] = 0
    idx[1:69] = 2
    idx[69:71] = 7
    idx[71:73] = 8
    idx[73:75] = 9
    idx[75:77] = 10
    idx[77:79] = 14
    idx[79:81] = 15
    idx[81:83] = 16
    idx[83:85] = 17
    idx[85:87] = 1
    idx[87] = 3
    idx[88] = 4
    idx[89] = 3
    idx[90] = 4
    idx[91] = 11
    idx[92] = 12
    idx[93] = 0
    idx[94] = 13
    idx[95:97] = 5
    idx[97:99] = 6

    return idx[label.long()]
```

- Verify that label map matches Synthseg provided labels
- Plug into training loop
- Eval on epoch / Eval after train (i'm leaning on latter)
