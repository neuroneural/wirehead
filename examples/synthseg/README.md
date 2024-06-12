# Wirehead + Synthseg Local Example

```
https://github.com/BBillot/SynthSeg
```

## Installation 

Instructions:
```
git clone git@github.com:neuroneural/wirehead.git
cd wirehead/examples/synthseg
```

Install python3.8 and create an environment
```
sudo apt install python3.8 python3.8-venv
python3 -m venv wirehead 
source wirehead/bin/activate
```

Install wirehead
```
pip install -e ../../
pip install -r requirements.txt
```

Fetch the data from SynthSeg
```
git clone git@github.com:BBillot/SynthSeg.git
```

Run the test
```
chmod +x test.sh
./test.sh
```

