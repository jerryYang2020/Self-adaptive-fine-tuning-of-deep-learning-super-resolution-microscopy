import gradio as gr
from unit.Run_demo_for_SFSRM_WSF import runFineTuning,runSuperResolutionRecon
from unit.Run_demo_for_SFSRM_SSF import runSSFSuperResolutionRecon, runSSFFineTuning

def run():
    with gr.Blocks(css='style\style.css') as demo:
        with gr.Row():
            with gr.Column(scale=15):
                gr.Markdown(value="<h1 id='h1'> Online Fine-tuning of Deep-learning Super-resolution Microscopy</h1>")
                gr.Markdown(value="<h2 id='h2'>[GitHub](https://github.com/jerryYang2020/Online_fine-tuning_of_deep-learning_super-resolution_microscopy.git) </h2>")
                gr.Markdown(value="<h2 id='h2'> If this method is helpful for you, please cite our paper. Thanks!</h2>")
        with gr.Tab(label = "SSF"):
            with gr.Row():
                with gr.Column(scale=8):
                    with gr.Column():
                        lrImg = gr.Text(value='test_data\KDEL\KDEL_TIRF.tif', label="LR_Img")
                        edgeImg = gr.Text(value='test_data\KDEL\KDEL_EDGE.tif', label="Edge_Img")
                        datadir_save = gr.Text(value='test_data\KDEL\KDEL_SSF', label="ResultDir")
                        SFSRMmodeldir = gr.Text(value = 'Pretrained_model\SFSRM_SRmodel_KDEL.pth', label="Super-Resolution model")
                        DENmodeldir = gr.Text(value = 'Pretrained_model\Denoisingmodel_KDEL.pth', label="Denosing model")

                    with gr.Column():
                        with gr.Row():
                            scale = gr.Number(value=8, label="scale", interactive=True)
                            timelen_WSF = gr.Number(value=10, label="timelen", interactive=True)
                            iteration_times_WSF = gr.Number(value=3, label="iteration_times", interactive=True)
                        with gr.Row():
                            SRlr_WSF = gr.Number(value=0.00001, label="SRlr", interactive=True)
                            SRlr_sigma = gr.Number(value=1, label="sigma", interactive=True)
                            SRlr_ksize = gr.Number(value=5, label="ksize", interactive=True)
                        with gr.Row():
                            continuity_level = gr.Number(value=1, label="continuity_level", interactive=True)
                            sparse_level = gr.Number(value=1, label="sparse_level", interactive=True)
                            consistency_level = gr.Number(value=2, label="consistency_level", interactive=True)
                    with gr.Column():
                        with gr.Row():
                            step = gr.Number(value=40, label="step", interactive=True)
                            gap = gr.Number(value=2, label="gap", interactive=True)
                with gr.Column(scale=15):
                    fineTuning_img = gr.Gallery(type='pil',scale=15)
                    FineTuning = gr.Button(value="FineTuning", variant="primary")
                    FineTuning.click(runSSFFineTuning, inputs=[lrImg,
                                                               edgeImg,
                                                               datadir_save,
                                                               SFSRMmodeldir,
                                                               DENmodeldir,
                                                               scale,
                                                               timelen_WSF,
                                                               iteration_times_WSF,
                                                               SRlr_sigma,
                                                               SRlr_ksize,
                                                               SRlr_WSF,
                                                               continuity_level,
                                                               sparse_level,
                                                               consistency_level,
                                                               step,
                                                               gap], outputs=fineTuning_img)
                    output_image = gr.PlayableVideo(format='mp4', scale=15)
                    with gr.Row():
                        index = gr.Number(value=3, label="iteration times", interactive=True)
                        SuperResolution = gr.Button(value="SuperResolution", variant="primary")
                        SuperResolution.click(runSSFSuperResolutionRecon, inputs=[index,lrImg,edgeImg,datadir_save], outputs=output_image)
        with gr.Tab(label = "WSF"):
            with gr.Row():
                with gr.Column(scale=8):
                    with gr.Column():
                        lrImg = gr.Text(value='test_data\KDEL\KDEL_TIRF.tif', label="LR_Img")
                        edgeImg = gr.Text(value='test_data\KDEL\KDEL_EDGE.tif', label="Edge_Img")
                        SMLMImg = gr.Text(value='test_data\KDEL\KDEL_SMLM.tif', label="SMLM_Img")
                        datadir_save = gr.Text(value='test_data\KDEL\KDEL_WSF', label="ResultDir")
                        SFSRMmodeldir = gr.Text(value = 'Pretrained_model\SFSRM_SRmodel_KDEL.pth', label="Super-Resolution model")
                    with gr.Column():
                        with gr.Row():
                            scale = gr.Number(value=8, label="scale", interactive=True)
                            timelen_WSF = gr.Number(value=10, label="timelen", interactive=True)
                        with gr.Row():
                            iteration_times_WSF = gr.Number(value=3, label="iteration_times", interactive=True)
                            SRlr_WSF = gr.Number(value=0.00001, label="SRlr", interactive=True)
                    with gr.Column():
                        with gr.Row():
                            step = gr.Number(value=40, label="step", interactive=True)
                            gap = gr.Number(value=2, label="gap", interactive=True)
                with gr.Column(scale=15):
                    fineTuning_img = gr.Gallery(type='pil',scale=15)
                    FineTuning = gr.Button(value="FineTuning", variant="primary")
                    FineTuning.click(runFineTuning, inputs=[lrImg,
                                                            edgeImg,
                                                            SMLMImg,
                                                            datadir_save,
                                                            SFSRMmodeldir,
                                                            scale,
                                                            timelen_WSF,
                                                            iteration_times_WSF,
                                                            SRlr_WSF,
                                                            step,
                                                            gap], outputs=fineTuning_img)
                    output_image = gr.PlayableVideo(format='mp4', scale=15)
                    with gr.Row():
                        index = gr.Number(value=3, label="iteration times", interactive=True)
                        SuperResolution = gr.Button(value="SuperResolution", variant="primary")
                        SuperResolution.click(runSuperResolutionRecon, inputs=[index,lrImg, edgeImg, SMLMImg,datadir_save], outputs=output_image)
    demo.launch()
