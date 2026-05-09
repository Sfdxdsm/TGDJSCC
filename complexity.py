import torch
import models_jscc
from  models_jscc import Net
import models_classifier
import models_od
import models_sr
import thop
import time

# encoder_jscc = models_jscc.__dict__['enc_base_patch2_embed256'](C=64, selected_nodes_num=16,
#                                              window_size=4,
#                                              multiple_snr='1,4,7,10,13', chan_type='awgn',
#                                              latent_tokens_num=15)
#
# decoder_jscc = models_jscc.__dict__['dec_base_patch2_embed256'](C=64, selected_nodes_num=16)
# model_jscc = Net(encoder=encoder_jscc, decoder=decoder_jscc,
#             multiple_snr='1,4,7,10,13', chan_type='awgn')
# model_jscc.to('cuda:1')
#
# flops, params = thop.profile(model_jscc, inputs=(torch.randn((1, 3, 32, 32)).to('cuda:1'),))
# print(f"FLOPs: {flops / 1e9} G")
# print(f"Params: {params / 1e6} M")
#
# model_jscc.eval()
# model_jscc.to('cuda:1')
# input_data = torch.randn((1, 3, 32, 32)).to('cuda:1')
# start_time = time.time()
# output = model_jscc(input_data)
# end_time = time.time()
# print(f"Execution time: {end_time - start_time}")

model_classification = models_classifier.__dict__['classifier_base_patch2_embed256'](
    drop_path_rate=0.2,
    n_classes=100,
    multiple_snr='1,4,7,10,13',
    chan_type='awgn',
    C=64,
    selected_nodes_num=16,
    window_size=4,
    latent_tokens_num=0
)

model_classification.to('cuda:1')

flops, params = thop.profile(model_classification, inputs=(torch.randn((1, 3, 32, 32)).to('cuda:1'),))
print(f"FLOPs: {flops / 1e9} G")
print(f"Params: {params / 1e6} M")

model_classification.eval()
model_classification.to('cuda:1')
input_data = torch.randn((1, 3, 32, 32)).to('cuda:1')
start_time = time.time()
output = model_classification(input_data)
end_time = time.time()
print(f"Execution time: {end_time - start_time}")

# encoder_detection = models_od.__dict__['enc_base_patch2_embed256'](drop_path_rate=0.2, C=64,
#                                            selected_nodes_num=16, window_size=4,
#                                            multiple_snr='1,4,7,10,13', chan_type='awgn',
#                                            latent_tokens_num=0)
#
# decoder_detection = models_od.__dict__['dec_base_patch2_embed256'](C=64, selected_nodes_num=16,
#                                            num_queries=20, num_classes=100)
# model_detection = models_od.Net(encoder=encoder_detection, decoder=decoder_detection, bbox_loss_coef=1,
#                       giou_loss_coef=1, set_cost_class=1,
#                       set_cost_box=1, set_cost_giou=1, eos_coef=1,
#                       multiple_snr='1,4,7,10,13', chan_type='awgn')
# model_detection.to('cuda:1')
#
# flops, params = thop.profile(model_detection, inputs=(torch.randn((1, 3, 32, 32)).to('cuda:1'),))
# print(f"FLOPs: {flops / 1e9} G")
# print(f"Params: {params / 1e6} M")
#
# model_detection.eval()
# model_detection.to('cuda:1')
# input_data = torch.randn((1, 3, 32, 32)).to('cuda:1')
# start_time = time.time()
# output = model_detection(input_data)
# end_time = time.time()
# print(f"Execution time: {end_time - start_time}")

# encoder = models_sr.__dict__['enc_base_patch2_embed256'](C=64, selected_nodes_num=16,
#                                              window_size=4,
#                                              multiple_snr='1,4,7,10,13', chan_type='awgn',
#                                              latent_tokens_num=15)
#
# decoder = models_sr.__dict__['dec_base_patch2_embed256'](C=64, selected_nodes_num=16,
#                                            scale_factor=2)
# model = models_sr.Net(encoder=encoder, decoder=decoder,
#             multiple_snr='1,4,7,10,13', chan_type='awgn')
# model.to('cuda:1')
# input_image = torch.randn((1, 3, 32, 32)).to('cuda:1')
# HR_image = torch.randn((1, 3, 64, 64)).to('cuda:1')
# flops, params = thop.profile(model, inputs=(input_image, HR_image, 10))
# print(f"FLOPs: {flops / 1e9} G")
# print(f"Params: {params / 1e6} M")
#
# model.eval()
# model.to('cuda:1')
# input_data = torch.randn((1, 3, 32, 32)).to('cuda:1')
# start_time = time.time()
# output = model(input_data, HR_image)
# end_time = time.time()
# print(f"Execution time: {end_time - start_time}")