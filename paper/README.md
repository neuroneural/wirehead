here's what I'm running:
- 10k samples on naive synthseg
- 10k samples on synthseg + wirehead
- 100k samples on 10x synthseg + wirehead, with 10 different ground truth samples

hyperparams:
- everything is the same as original synthseg  paper

eval:
- faster_dice on train on every batch
- faster_dice on eval on every epoch


- fetch an input, label104 pair
- apply the function I shared yesterday to label104 to get label18
- run your synthseg trained model on input and produce predicted_label18
- apply faster_dice to label18  and predicted predicted_label18
