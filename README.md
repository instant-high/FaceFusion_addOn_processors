# FaceFusion addOn processors
### FaceFusion 3.4.0 - 3.5.0  
Additional minimal 'one-script' processors  
No installation, just drag'n'drop  

Download and unzip/copy each script to:  
**'facefusion/processors/modules/'**  

If the processor uses a pretrained onnx-model, download the model and it's hash file from 'releases'  
and unzip/copy to:  
**'.assets/models/'**

Activate/deactivate the processor under FaceFusion 'Processors' tab  
(no other settings)  

**- Auto White Balance** - based on the von Kries model (edit the script line 57 to use different profiles)  

**- Cartoonizer** - comic style effect:

https://github.com/user-attachments/assets/ecaaa776-5a12-47f4-908e-42c27b8db7e0  

.

**- Pencil** - black and white pencil drawing:  

https://github.com/user-attachments/assets/d1b416e2-872f-4237-991e-0d56639aef29  

.

**- DepthMap/2D to 3D**   
By default this processor creates a depth map as output  
You can add the following lines to 'facefusion.ini' to get Anaglyph, Half SBS or Half SBS crosseye output.  
(Download depth_anything_vits14.zip from releases)
 

[depth]  
3D_mode = anaglyph  

available options so far:  
anaglyph / half_sbs / half_sbs_crosseye  
If you enter none of that options, output will be depthmap  

https://github.com/user-attachments/assets/545a17c2-d6e2-4cf3-b0ea-c75a7c1186d0  

https://github.com/user-attachments/assets/8482e2a9-daeb-4e82-905b-12a0d569e6ec

https://github.com/user-attachments/assets/af113755-dff8-4aab-a8f7-4dacc9d4859f










