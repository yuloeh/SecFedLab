def create_trigger_fn(trigger_size=3, position='bottom_right'):
    def trigger_fn(img):
        # img 形状: (C, H, W)
        t = img.clone()
        _, H, W = t.shape
        if position == 'bottom_right':
            r0, c0 = H - trigger_size, W - trigger_size
        else:
            r0, c0 = 0, 0
        # 假设图像已做 (0.1307,), (0.3081,) 归一化，白色方块（原始值约1.0）的归一化值约为 2.82
        t[:, r0:r0+trigger_size, c0:c0+trigger_size] = 2.82
        return t
    return trigger_fn
