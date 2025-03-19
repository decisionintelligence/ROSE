# ReadyTS
# Readme


### Install requirements

```jsx
pip install -r requirements.txt
```

### Prepare dataset
Create a seperate folder **./data** and put all the csv dataset files in the directory.

### Finetune

The script under ***script/finetune*** folder is for fine-tuning. Either linear_probing or fine-tune the entire network can be applied.

```jsx
sh script/finetune/ETTh1.sh
```


### Zero-shot

The script under ***script/zero_shot*** folder is for zero-shot.

```jsx
sh script/zero_shot/exp_zero_shot.sh
```
