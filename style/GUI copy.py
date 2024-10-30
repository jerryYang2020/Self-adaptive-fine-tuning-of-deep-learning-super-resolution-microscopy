import gradio as gr
from unit.Run_demo_for_SFSRM_WSF import runFineTuning,runSuperResolutionRecon
from unit.Run_demo_for_SFSRM_SSF import runSSFSuperResolutionRecon, runSSFFineTuning

def run():
    with gr.Blocks(css='style\style.css') as demo:
        with gr.Row():
            with gr.Column(scale=15):
                gr.Markdown(value="<h1 id='h1'> Online Fine-tuning of Deep-learning Microscopy</h1>")
                gr.Markdown(value="<h2 id='h2'>[GitHub](https://www.baidu.com) | [Paper](https://www.baidu.com) | [Project Page](https://www.baidu.com)</h2>")
                gr.Markdown(value="<h2 id='h2'> If this method is helpful for you, please cite our paper. Thanks!</h2>")
        with gr.Tab(label = "SSF"):
            with gr.Row():
                with gr.Column(scale=8):
                    with gr.Column():
                        lrImg = gr.Text(value='test_data\SEC61B_TIRF.tif', label="Please enter the path to the LR image (relative to the project directory)")
                        edgeImg = gr.Text(value='test_data\SEC61B_EDGE.tif', label="Please enter the path to the Edge image (relative to the project directory)")
                        datadir_save = gr.Text(value='test_data\SEC61B_SSF', label="Result save path (relative to the project directory)")
                        SFSRMmodeldir = gr.Text(value = 'Pretrained_model\SFSRM_SRmodel_SEC61B.pth', label="Please enter the path of the pre-trained super-resolution model (relative to the project directory)")
                        DENmodeldir = gr.Text(value = 'Pretrained_model\Denoisingmodel_SEC61B.pth', label="Please enter the path where the pre-trained denoising model is located (relative to the project directory)")

                    with gr.Column():
                        with gr.Row():
                            scale = gr.Number(value=8, label="upsampling scale of the Super-resolution reconstructions", interactive=True)
                            timelen_WSF = gr.Number(value=10, label="image len for training SFSRM (live-cell video)(default 10)", interactive=True)
                            iteration_times_WSF = gr.Number(value=5, label="iteration times of the SFSRM-SSF (default 20)", interactive=True)
                        with gr.Row():
                            SRlr_WSF = gr.Number(value=0.00001, label="learning rate for the fine-tuning of SFSRM-WSF", interactive=True)
                            SRlr_sigma = gr.Number(value=1.6, label="Gaussian sigma of the RSF before upsampling (default 1.6)", interactive=True)
                            SRlr_ksize = gr.Number(value=5, label="kernal size of the simulated RSF (default 5)", interactive=True)
                        with gr.Row():
                            continuity_level = gr.Number(value=1, label="level of the continuity regularization term (default 1)", interactive=True)
                            sparse_level = gr.Number(value=1, label="level of the sparse regularization term (default 1.5)", interactive=True)
                            consistency_level = gr.Number(value=2, label="level of the consistency regularization term (default 1)", interactive=True)
                    with gr.Column():
                        with gr.Row():
                            step = gr.Number(value=40, label="step length for super-resolution reconstructions", interactive=True)
                            gap = gr.Number(value=2, label="gap size for super-resolution reconstructions", interactive=True)
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
                        index = gr.Number(value=1, label="iteration times", interactive=True)
                        SuperResolution = gr.Button(value="SuperResolution", variant="primary")
                        SuperResolution.click(runSSFSuperResolutionRecon, inputs=[index,lrImg, edgeImg], outputs=output_image)
        with gr.Tab(label = "WSF"):
            with gr.Row():
                with gr.Column(scale=8):
                    with gr.Column():
                        lrImg = gr.Text(value='test_data\SEC61B_TIRF.tif', label="Please enter the path to the LR image (relative to the project directory)")
                        edgeImg = gr.Text(value='test_data\SEC61B_EDGE.tif', label="Please enter the path to the Edge image (relative to the project directory)")
                        SMLMImg = gr.Text(value='test_data\SEC61B_SMLM.tif', label="Please enter the path to the SMLM image (relative to the project directory)")
                        datadir_save = gr.Text(value='test_data\SEC61B_WSF', label="Result save path (relative to the project directory)")
                        SFSRMmodeldir = gr.Text(value = 'Pretrained_model\SFSRM_SRmodel_SEC61B.pth', label="Please enter the path where the pre-trained model is located (relative to the project directory)")
                    with gr.Column():
                        with gr.Row():
                            scale = gr.Number(value=8, label="upsampling scale of the Super-resolution reconstructions", interactive=True)
                            timelen_WSF = gr.Number(value=10, label="image len for training SFSRM-WSF", interactive=True)
                        with gr.Row():
                            iteration_times_WSF = gr.Number(value=1, label="iteration times of the SFSRM-WSF", interactive=True)
                            SRlr_WSF = gr.Number(value=0.00001, label="learning rate for the fine-tuning of SFSRM-WSF", interactive=True)
                    with gr.Column():
                        with gr.Row():
                            step = gr.Number(value=40, label="step length for super-resolution reconstructions", interactive=True)
                            gap = gr.Number(value=2, label="gap size for super-resolution reconstructions", interactive=True)
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
                        index = gr.Number(value=1, label="iteration times", interactive=True)
                        SuperResolution = gr.Button(value="SuperResolution", variant="primary")
                        SuperResolution.click(runSuperResolutionRecon, inputs=[index,lrImg, edgeImg, SMLMImg], outputs=output_image)
    demo.launch()
