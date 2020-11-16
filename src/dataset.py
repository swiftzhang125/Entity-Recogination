import torch

class EntityDataset():
    def __init__(self, words, pos, tags, tokenizer, max_len):
        self.words = words
        self.pos = pos
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.words)

    def __getitem__(self, item):
        words = self.words[item]
        pos = self.pos[item]
        tags = self.tags[item]

        ids = []
        tar_pos = []
        tar_tag = []

        for i, w in enumerate(words):
            inputs = self.tokenizer.encode(
                w,
                add_special_tokens=False,
            )

            # swiftzhang[mask] : 'sw##'[mask1] + '##if##'[mask2] + '##zhang'[mask3]
            input_len = len(inputs)
            ids.extend(inputs)
            tar_pos.extend([pos[i]] * input_len)
            tar_tag.extend([tags[i]] * input_len)

        ids = ids[: (self.max_len - 2)]
        tar_pos = tar_pos[: (self.max_len - 2)]
        tar_tag = tar_tag[: (self.max_len - 2)]

        # [CLS] : [101]
        # [SP] : [102] . / ? !
        # 【mask】 : [1 - 10,000,000]
        ids = [101] + ids + [102]
        tar_pos = [0] + tar_pos + [0]
        tar_tag = [0] + tar_tag + [0]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = self.max_len - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        tar_pos = tar_pos + ([0] * padding_len)
        tar_tag = tar_tag + ([0] * padding_len)


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target_pos': torch.tensor(tar_pos, dtype=torch.long),
            'target_tag': torch.tensor(tar_tag, dtype=torch.long)
        }



