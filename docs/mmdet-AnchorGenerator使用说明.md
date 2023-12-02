## AnchorGenerator使用说明

### 文件位置: 
    mmdet/models/task_modules/prior_generators/anchor_generator.py

生成base_anchors需要3个参数scales, ratios, base_sizes

**scales:**
    该参数表示对anchor_base_size的缩放系数
    该参数通过"scales"指定, 或者通过"octave_base_scale"、"scales_per_octave"两个参数计算得出, 但两套参数不能同时设定

**ratios:**
    表示每个anchor的 h / w

**base_sizes:**
    表示每层feature map的anchor的base_size
    该参数可通过"base_sizes"来设置，也可以通过"strides"参数来设置, 当"base_sizes"为None时, 默认使用"strides"

最终每个grid会生成len(scales) * len(ratios)个anchor


**gen_base_anchors()方法**
    用于生成每层feature-map的anchor尺寸, 每层anchor的维度为[num_anchor_per_grid, 4]

**grid_priors()方法**
    对于生成每层feature-map每个grid的anchor, 维度为[num_grid_per_feature * num_anchor_per_grid, 4]

**valid_flags()**方法
    根据img_meta['pad_shape']中的尺寸, 计算在不同下采样(stride)倍数下对应的feature-map哪些grid是有效grid(是为了防止在conv过程中padding导致feature-map的尺寸与实际下采样尺寸不统一的问题？)