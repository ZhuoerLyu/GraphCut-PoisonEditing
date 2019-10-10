%% read images
source_image =im2double(imread('ECE7866HW2_upload/Assignment_2_code/data/Poisson_editing/data2/foreground.jpg'));
target_image = im2double(imread('ECE7866HW2_upload/Assignment_2_code/data/Poisson_editing/data2/background.jpg'));
mask_image = rgb2gray(im2double(imread('ECE7866HW2_upload/Assignment_2_code/data/Poisson_editing/data2/mask.png')));
mask_image = ~mask_image;
% mask_image = mask_image(:,:,1);

%expand source image and mask image to the size of target image
height = size(target_image,1);
width = size(target_image,2);
%mask image
new_mask = zeros([height width]);
new_mask(1:510,1:825) = mask_image;
mask_image = new_mask;
%source image
new_source = zeros(size(target_image));
for i = 1:3
    new_source(1:510,1:825,i) = source_image(:,:,i);
end
source_image = new_source;
figure;
imshow(source_image);
figure;
imshow(mask_image);

%% generate b
%calculate graident of source image
b = bsxfun(@times,target_image, repmat((mask_image),[1 1 3]));
 %generate gradient part for 
source_gradient_image = bsxfun(@times, source_image, repmat((mask_image),[1 1 3]));
% img_gradient_r = conv2(source_gradient_image(:,:,1), laplacian_kernel,'same');
% img_gradient_g = conv2(source_gradient_image(:,:,2), laplacian_kernel,'same');
% img_gradient_b = conv2(source_gradient_image(:,:,3), laplacian_kernel,'same');

%Qianlong image gradient code
La=fspecial('laplacian',0);

img_gradient=imfilter(source_image,La,'replicate');

img_gradient_r = img_gradient(:,:,1);
img_gradient_g = img_gradient(:,:,2);
img_gradient_b = img_gradient(:,:,3);
figure;
imshow(source_gradient_image);
figure;
imshow(img_gradient);


%separate each channel


%% generate sparse matrix
height = size(mask_image,1);
width = size(mask_image,2);
%size of array
num_of_1 = sum(sum(mask_image ~= 0));
num_of_0 = sum(sum(mask_image == 0));
total_dimension = num_of_0 + num_of_1 * 5;
%element for ii jj value
ii = zeros(total_dimension,1);
jj = zeros(total_dimension, 1);
value = zeros([total_dimension 1]);
b = zeros(height*width,3);


count = 1;
for row = 1 : height ;
    for col = 1 : width ;
        pixel =  (row - 1)*width + col;
        if mask_image(row, col) == 0;
            ii(count,1) = pixel;
            jj(count,1) = pixel;
            value(count,1) =1;
            count = count+1;
            %when the pixel is out out of the mask, just use the source
            %image value
            b(pixel,1) = target_image(row,col,1);
            b(pixel,2) = target_image(row,col,2); 
            b(pixel,3) = target_image(row,col,3); 

        else
            %when the pixel is in the mask, use the gradient image label
            b(pixel,1) = img_gradient_r(row,col);
            b(pixel,2) = img_gradient_g(row,col); 
            b(pixel,3) = img_gradient_b(row,col); 
            
            ii(count,1) = pixel;
            jj(count,1) = pixel;
            value(count,1) = -4;
            count = count+1;
            %up
            ii(count,1) = pixel;
            jj(count,1)= col+(row-2)*width;
            value(count,1) = 1;
            count = count+1;

            %A(pixel, 1+col+(row+1)*width) = 1; end
            %left
            ii(count,1) = pixel;
            jj(count,1) = col+(row-1)*width-1;
            value(count,1) =1; 
            count = count+1;

            % A(pixel, 1+col+(row-1)*width) = 1
            %right
            ii(count,1) = pixel;
            jj(count,1) = col+1+(row-1)*width;
            value(count,1) =1; 
            count = count+1

            %A(pixel, 1+(col+1)+row*width) = 1; end
            %down
            ii(count,1) = pixel;
            jj(count,1) = col+row*width;
            value(count,1) =1;
            %A(pixel, 1+(col-1)+row*width) = 1; 
            count = count+1;

      
        end
    end
end

A = sparse(ii,jj,value);

%% 

generated_image_r = A\b(:,1);
generated_image_g = A\b(:,2);
generated_image_b = A\b(:,3);


image_r = reshape(generated_image_r,[width height])';
image_g = reshape(generated_image_g,[width height])';
image_b = reshape(generated_image_b,[width height])';
final_image = zeros([height width 3]);
final_image(:,:,1) = image_r;
final_image(:,:,2) = image_g;
final_image(:,:,3) = image_b;
figure;
imshow(final_image);
imwrite(final_image,'result.png');