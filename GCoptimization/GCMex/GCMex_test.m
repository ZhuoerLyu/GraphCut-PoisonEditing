
W = 512;
H = 384;
num_pixel = 384*512;
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

[labels E Eafter] = GCMex(segclass_own, single(probclass), pairwise, single([0 1; 1 0]),0);

fprintf('E: %d (should be 260), Eafter: %d (should be 44)\n', E, Eafter);
fprintf('unique(labels) should be [0 4] and is: [');
fprintf('%d ', unique(labels));
fprintf(']\n');
