# Gomoku RL Training On Google Colab

Huong dan nay dung cho file `colab_train.py` trong project nay.

## 1. Chuan bi tren may local

- Dam bao file `colab_train.py` da o root project.
- Neu muon resume sau nay, giu lai thu muc `models/` duoc download tu session truoc.

## 2. Tao notebook moi tren Google Colab

1. Dang nhap [colab.research.google.com](https://colab.research.google.com).
2. Chon **File -> New notebook**.
3. Vao **Runtime -> Change runtime type**.
4. Chon **Hardware accelerator: GPU** (T4, P100, L4 deu dung duoc).
5. Dat ten notebook, vi du: `gomoku-rl-training`.

## 3. Upload file can thiet

1. O panel **Files** ben trai, chon **Upload to session storage**.
2. Upload file `colab_train.py`.
3. Sau khi upload, file se nam o duong dan:

```text
/content/colab_train.py
```

Neu ban muon resume training tu checkpoint cu, upload them 3 file sau vao thu muc `models/` trong session moi:

- `rl_agent.pth`
- `rl_agent_buffer.npz`
- `checkpoint.json`

## 4. Cai thu vien

Tao mot cell moi va chay:

```python
!pip install numpy pandas torch --quiet
```

## 5. Chay training lan dau

Tao cell tiep theo va chay:

```python
%run /content/colab_train.py --episodes 2000 --save-every 200
```

Y nghia nhanh:

- `--episodes 2000`: train 2000 van self-play trong session nay
- `--save-every 200`: luu checkpoint moi 200 van

## 6. Resume training tu checkpoint

Neu session cu da train mot phan va ban da upload lai checkpoint, chay:

```python
%run /content/colab_train.py --episodes 2000 --save-every 200 --resume
```

Voi `--resume`, script se:

- load model tu `models/rl_agent.pth`
- load replay buffer tu `models/rl_agent_buffer.npz`
- load metadata tu `models/checkpoint.json`
- tiep tuc dem tu `episodes_done + 1`

## 7. Cac tham so CLI ho tro

```text
--episodes
--batch-size
--buffer-size
--save-every
--epsilon-start
--epsilon-end
--model-path
--resume
```

Vi du chay voi tham so day du hon:

```python
%run /content/colab_train.py --episodes 3000 --batch-size 64 --buffer-size 100000 --save-every 200 --epsilon-start 0.5 --epsilon-end 0.01
```

## 8. File output sau khi train

Sau khi train, Colab se tao cac file sau:

```text
/content/models/rl_agent.pth
/content/models/rl_agent_buffer.npz
/content/models/checkpoint.json
/content/logs/replays.jsonl
```

Y nghia:

- `rl_agent.pth`: trong so model
- `rl_agent_buffer.npz`: replay buffer de resume
- `checkpoint.json`: so episode da train va thong ke win/draw
- `replays.jsonl`: lich su van dau

## 9. Download file ve may

1. Trong panel **Files**, mo thu muc `models/`.
2. Download it nhat file `rl_agent.pth`.
3. Neu muon resume sau nay, download them:
   - `rl_agent_buffer.npz`
   - `checkpoint.json`

## 10. Dung model da train o local

Dat file model vao thu muc `models/` trong project local, sau do chay:

```bash
python src/scripts/compare.py --matches 20 --rl-model models/rl_agent.pth
```

Neu muon mo GUI voi RL model:

```bash
python src/main.py --mode aivai --depth 3 --rl-model models/rl_agent.pth
```

Luu y: hien tai `--rl-model` duoc dung trong mode `aivai`.

## 11. Goi y cau hinh

- Chay thu nhanh: `--episodes 500`
- Chay muc co y nghia: `--episodes 2000` den `3000`
- Luu checkpoint an toan: `--save-every 100` hoac `200`
- Neu session Colab sap reset, download checkpoint de resume sau

## 12. Luu y thuc te

- Colab co gioi han thoi gian session, thuong 12h va co the thay doi theo tai khoan.
- GPU nhanh hon CPU rat nhieu cho self-play + PyTorch.
- Day la board 9x9, nen model phai duoc train moi tu dau.
- `Validation vs random` tang dan la dau hieu tot, nhung can dung `compare.py` de danh gia ky hon voi Minimax.
