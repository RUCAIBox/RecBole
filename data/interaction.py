# @Time   : 2020/7/10
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time    : 2020/8/6, 2020/8/6
# @Author  : Yupeng Hou, Yushuo Chen
# @email   : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn


class Interaction(object):
    def __init__(self, interaction, pos_len_list=None, user_len_list=None):
        self.interaction = interaction
        self.pos_len_list = pos_len_list
        self.user_len_list = user_len_list
        for k in self.interaction:
            self.length = self.interaction[k].shape[0]
            break

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.interaction[index]
        else:
            ret = {}
            for k in self.interaction:
                ret[k] = self.interaction[k][index]
            return Interaction(ret)

    def __len__(self):
        return self.length

    def __str__(self):
        info = ['The batch_size of interaction: {}'.format(self.length)]
        for k in self.interaction:
            temp_str = "    {}, {}, {}".format(k, self.interaction[k].shape, self.interaction[k].device.type)
            info.append(temp_str)
        info.append('\n')
        return '\n'.join(info)


    def to(self, device, selected_field=None):
        ret = {}
        try:
            selected_field = set(selected_field)
            for k in self.interaction:
                if k in selected_field:
                    ret[k] = self.interaction[k].to(device)
                else:
                    ret[k] = self.interaction[k]
        except:
            for k in self.interaction:
                ret[k] = self.interaction[k].to(device)
        return Interaction(ret)

    def cpu(self):
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].cpu()
        return Interaction(ret)

    def numpy(self):
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].numpy()
        return Interaction(ret)

    def repeat(self, sizes):
        ret = {}
        for k in self.interaction:
            if len(self.interaction[k].shape) == 1:
                ret[k] = self.interaction[k].repeat(sizes)
            else:
                ret[k] = self.interaction[k].repeat([sizes, 1])
        return Interaction(ret)

    def to_device_repeat_interleave(self, device, repeats, dim=0):
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].to(device).repeat_interleave(repeats, dim=dim)
        return Interaction(ret)

    def update(self, new_inter):
        for k in new_inter.interaction:
            self.interaction[k] = new_inter.interaction[k]
