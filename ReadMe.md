ImageConvolution application performs convolve operator on a given image.

Qt is used to design the front end of the application.

CUDA handles the convolution operation.

	To peform convolution on a given image, carry out the below steps.
	1) Click File->Open to open an image.
	2) Click Convolve->Set Kernel to set the kernel for convolve operator
	3) Click Convolve->Apply Kernel to convolve the input image with supplied kernel.
	Once step 3 is carried out, the resultant convolved image will be shown.
	
    You should be able to see the run times of each image on the console window that opened along with the GUI window when
    the application started.
