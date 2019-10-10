%read image
stroke = imread('ECE7866HW2_upload/Assignment_2_code/data/Lazysnapping_data/dog stroke.png');
original_image =imread('ECE7866HW2_upload/Assignment_2_code/data/Lazysnapping_data/dog.PNG');
imshow(stroke);
imshow(original_image)
%separate rgb channels
r_channel = stroke(:,:,1);
g_channel = stroke(:,:,2);
b_channel = stroke(:,:,3);
%plot r and b channel to check 
fore_image = bsxfun(@times, original_image, repmat(r_channel/max(max(r_channel)),[1,1,3]));
figure;
imshow(fore_image);

back_image = bsxfun(@times, original_image, repmat(b_channel/max(max(b_channel)),[1,1,3]));
figure;
imshow(back_image);
%% find the fore point  and background point
k = find(rgb2gray(fore_image));
fore_point_1 = fore_image(:,:,1);
fore_point_1 = fore_point_1(k);
fore_point_2 = fore_image(:,:,2);
fore_point_2 = fore_point_2(k);
fore_point_3 = fore_image(:,:,3);
fore_point_3 = fore_point_3(k);

fore_point = zeros(size(k,1),3);
fore_point(:,1) = fore_point_1;
fore_point(:,2) = fore_point_2;
fore_point(:,3) = fore_point_3;

% find the fore point 
k = find(rgb2gray(back_image));
back_point_1 = back_image(:,:,1);
back_point_1 = back_point_1(k);
back_point_2 = back_image(:,:,2);
back_point_2 = back_point_2(k);
back_point_3 = back_image(:,:,3);
back_point_3 = back_point_3(k);

back_point = zeros(size(k,1),3);
back_point(:,1) = back_point_1;
back_point(:,2) = back_point_2;
back_point(:,3) = back_point_3;

%% use kmeans to cluster fore points and back ground points
% foreground point gaussian model
fore_point_cluster = kmeans(fore_point,2);
back_point_cluster = kmeans(back_point,2);

fore_point_gmm_1 = fitgmdist(fore_point(find(fore_point_cluster == 1),:),2);
fore_point_gmm_1 = gmdistribution(fore_point_gmm_1.mu,fore_point_gmm_1.Sigma,fore_point_gmm_1.ComponentProportion);

fore_point_gmm_2 = fitgmdist(fore_point(find(fore_point_cluster == 2),:),2);
fore_point_gmm_2 = gmdistribution(fore_point_gmm_2.mu,fore_point_gmm_2.Sigma,fore_point_gmm_2.ComponentProportion);

back_point_gmm_1 = fitgmdist(back_point(find(back_point_cluster == 1),:),2);
back_point_gmm_1 = gmdistribution(back_point_gmm_1.mu,back_point_gmm_1.Sigma,back_point_gmm_1.ComponentProportion);

back_point_gmm_2 = fitgmdist(back_point(find(back_point_cluster == 2),:),2);
back_point_gmm_2 = gmdistribution(back_point_gmm_2.mu,back_point_gmm_2.Sigma,back_point_gmm_2.ComponentProportion);

fore_gmm_mean = [fore_point_gmm_1.mu,fore_point_gmm_2.mu];
fore_gmm_variance = [fore_point_gmm_1.Sigma, fore_point_gmm_2.Sigma];
fore_gmm_coe = [fore_point_gmm_1.ComponentProportion,fore_point_gmm_2.ComponentProportion];
%% generate new image, seg label, 
new_cut_image = zeros(size(original_image));
segclass_own = zeros(1,size(original_image,1)*size(original_image,2));
probclass = zeros(2,size(original_image,1)*size(original_image,2));
count_segclass = 1
for row = 1 : size(original_image,1)
    for col =1: size(original_image,2)
        prob_fore_gmm_1 = mvnpdf(double(reshape(original_image(row,col,:),[1,3])),fore_point_gmm_1.mu(1,:),fore_point_gmm_1.Sigma(:,:,1));
        prob_fore_gmm_2 = mvnpdf(double(reshape(original_image(row,col,:),[1,3])),fore_point_gmm_2.mu(1,:),fore_point_gmm_2.Sigma(:,:,1));
        prob_back_gmm_1 = mvnpdf(double(reshape(original_image(row,col,:),[1,3])),back_point_gmm_1.mu(1,:),back_point_gmm_1.Sigma(:,:,1));
        prob_back_gmm_2 = mvnpdf(double(reshape(original_image(row,col,:),[1,3])),back_point_gmm_2.mu(1,:),back_point_gmm_2.Sigma(:,:,1));
        temp_prob_array = [prob_fore_gmm_1,prob_fore_gmm_2,prob_back_gmm_1,prob_back_gmm_2];
        index = find(temp_prob_array == max(temp_prob_array));
        if index == 1 | index == 2
            new_cut_image(row,col,:) = original_image(row,col,:);
            segclass_own(1,count_segclass) = 1; %denote as foreground
        else
            segclass_own(1,count_segclass) = 0; %denote as back ground
        %generating 
        end
        fore_max_prob = max(temp_prob_array(1:2));
        probclass(1,count_segclass) = fore_max_prob;
        back_max_prob = max(temp_prob_array(3:4));
        probclass(2,count_segclass) = back_max_prob;
        count_segclass = count_segclass +1;
            
        
    end
end


%% mean cut

H = size(original_image,1);
W = size(original_image,2);
num_pixel = W*H;
segclass = zeros(num_pixel,1);
pairwise = sparse(num_pixel,num_pixel);
unary = zeros(2,num_pixel/2);
[X Y] = meshgrid(1:2, 1:2);
labelcost = min(2, (X - Y).*(X - Y));

for row = 0:H-1
  for col = 0:W-1
    pixel = 1+ row*W + col;
    centroid_prob = original_image(row + 1, col + 1,:);
    if row+1 < H, pairwise(pixel, 1+col+(row+1)*W) = exp(-norm(reshape(double((original_image(row+1+1,col+1,:) - centroid_prob)),[1,3]).^2)); end %down
    if row-1 >= 0, pairwise(pixel, 1+col+(row-1)*W) = exp(-norm(reshape(double((original_image(row-1+1,col+1,:) - centroid_prob)),[1,3]).^2)); end %up
    if col+1 < W, pairwise(pixel, 1+(col+1)+row*W) = exp(-norm(reshape(double((original_image(row+1,col+1+1,:) - centroid_prob)),[1,3]).^2)); end % right
    if col-1 >= 0, pairwise(pixel, 1+(col-1)+row*W) = exp(-norm(reshape(double((original_image(row+1,col-1+1,:) - centroid_prob)),[1,3]).^2)); end %left
%     if pixel < 25
%       unary(:,pixel) = [0 10 10 10 10 10 10]'; 
%     else
%       unary(:,pixel) = [10 10 10 10 0 10 10]'; 
%     end
  end
end

[labels E Eafter] = GCMex(segclass_own, single(probclass), pairwise, single([0 4; 4 0]),0);

fprintf('E: %d (should be 260), Eafter: %d (should be 44)\n', E, Eafter);
fprintf('unique(labels) should be [0 4] and is: [');
fprintf('%d ', unique(labels));
fprintf(']\n');

%% generate new image
labels = reshape(labels, [W,H])';
figure;
imshow(labels);



new_image = bsxfun(@times, im2double(original_image), repmat(labels,[1,1,3]));
figure;
imshow(new_image);