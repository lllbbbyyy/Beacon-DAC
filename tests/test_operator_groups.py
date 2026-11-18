import unittest
from chiplet_tuner.core.utils import layer_group, operator_features


class OperatorGroupTests(unittest.TestCase):
    def test_compass_stage_names(self):
        examples = {
            "layer0_qkv_gen_fused_tp_shard6": "qkv_projection",
            "layer0_QKV_Gen": "qkv_projection",
            "layer0_q_gen_merged": "qkv_projection",
            "layer0_k_gen_merged": "qkv_projection",
            "layer0_v_gen_merged": "qkv_projection",
            "layer0_head0_Q": "qkv_projection",
            "layer0_head0_K": "qkv_projection",
            "layer0_head0_V": "qkv_projection",
            "layer0_Attn_QK": "attention",
            "layer0_Attn_AV": "attention",
            "layer0_head0_req1_QK_P": "attention",
            "layer0_head0_req1_QKV_D": "attention",
            "layer0_q_norm_head3": "other",
            "layer0_k_norm_head3": "other",
            "layer0_ffn2_tiling1": "ffn",
            "layer0_moe_expert0_down": "ffn",
            "layer0_out_proj_tiling1": "output_projection",
        }
        for name, expected in examples.items():
            with self.subTest(name=name):
                self.assertEqual(layer_group(name), expected)
                self.assertEqual(operator_features(name)["group"], expected)
