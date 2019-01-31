import numpy as np
import torch
from transformer_modules import make_std_mask
from transformer_model import make_model

print('0. Vocabulary')

fr_vocab = ['etudiant', 'je', 'quel', 'suis', 'mois', '<blank>']
en_vocab = ['a', 'am', 'month', 'i', 'student', 'what', '<blank>', '<s>', '</s>']

src_pad = 5
tgt_pad = 6

print('source vocabulary: {}'.format(fr_vocab))
print('target vocabulary: {}'.format(en_vocab))
print('')

print('1. Preparing batch and mask')
print('')
print('1-1. Batch (text)')

src_txt = [['je', 'suis', 'etudiant'],
        ['quel', 'mois', '<blank>']]

tgt_txt = [['<s>', 'i', 'am', 'a', 'student'],
        ['<s>', 'what', 'month', '</s>', '<blank>']]

tgt_txt_out = [['i', 'am', 'a', 'student', '</s>'],
        ['what', 'month', '</s>', '<blank>', '<blank>']]

print('source batch (text): {}'.format(src_txt))
print('target batch (text): {}'.format(tgt_txt))
print('target batch true output: {}'.format(tgt_txt_out))
print('')

print('1-2. Batch (number)')

src_num = []
tgt_num = []

for sen in src_txt:
    sen2num = []

    for word in sen:
        sen2num.append(fr_vocab.index(word))

    src_num.append(sen2num)

for sen in tgt_txt:
    sen2num = []

    for word in sen:
        sen2num.append(en_vocab.index(word))

    tgt_num.append(sen2num)

src_num = torch.tensor(np.array(src_num))
tgt_num = torch.tensor(np.array(tgt_num))

print('source batch (number):')
print(src_num)
print('')
print('target batch (number):')
print(tgt_num)
print('')

print('1-3. Mask')

src_mask = (src_num != src_pad).unsqueeze(-2)
tgt_mask = make_std_mask(tgt_num, tgt_pad)

print('source mask:')
print(src_mask)
print('')
print('target mask:')
print(tgt_mask)
print('')

model = make_model(len(fr_vocab), len(en_vocab), N=1, d_model=6, d_ff=24, h=3)

out = model(src_num, tgt_num, src_mask, tgt_mask)
print('out:')
print(out)
print('')

print('4. Prediction')
print('')

print('4-1. Calculate probability (linear map + softmax)')
print('')
print('target vocabulary: {}'.format(en_vocab))
prob = model.generator(out)

print('4-2. Predict based on the probability')
print('argmax:')
argmax_index = torch.max(prob, -1)[1]
print(argmax_index)
print('')

c = argmax_index.shape[0]
r = argmax_index.shape[1]

pred_sen = []

for i in range(c):
    tmp_sen = []
    for j in range(r):
        tmp_sen.append(en_vocab[argmax_index[i][j].item()])
    pred_sen.append(tmp_sen)

print('predicted sentences:')
print(pred_sen)
print('')

print('4-3. Compare to the true output')
print('target batch true output: {}'.format(tgt_txt_out))
