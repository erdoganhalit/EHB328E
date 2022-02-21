%read all pictures in the training/men folder to struct
men_training_folder = 'training/men/';
men_training_struct = dir([men_training_folder '*.jpg']);

%read all pictures in the training/women folder to struct
women_training_folder = 'training/women/';
women_training_struct = dir([women_training_folder '*.jpg']);

%convert struct to array and array to matrix for men training pictures
men_training_pics_array = [];
for k = 1:length(men_training_struct)
    men_pic_index = imread([men_training_folder men_training_struct(k).name]);
    men_pics_index = reshape(men_pic_index, 1, []);
    men_training_pics_array = [men_training_pics_array, men_pics_index]; 
end

%define number of samples(nSmp) and number of features(nFea) which is the
%same for both men and women
nSmp = 2500;
nFea = 1296;

%2500 is the number of pictures, 1296 is the number of pixels
men_training_pics = reshape(men_training_pics_array, [1296, 2500]);

%scale the features of men training pictures
men_training_pics = double(men_training_pics);
maxValue_m = max(max(men_training_pics));
men_training_pics = men_training_pics ./ maxValue_m;

%convert struct to array and array to matrix for women training pictures
women_training_pics_array = [];
for k = 1:length(women_training_struct)
    women_pic_index = imread([women_training_folder women_training_struct(k).name]);
    women_pics_index = reshape(women_pic_index, 1, []);
    women_training_pics_array = [women_training_pics_array, women_pics_index]; 
end


%2500 is the number of pictures, 1296 is the number of pixels
women_training_pics = reshape(women_training_pics_array, [1296, 2500]);

%scale the features of women training pictures
women_training_pics = double(women_training_pics);
maxValue_w = max(max(women_training_pics));
women_training_pics = women_training_pics ./ maxValue_w;

%subtrack mean men face
meanMen = mean(men_training_pics, 2);
men_training_pics = men_training_pics - repmat(meanMen, 1, nSmp);

%subtrack mean women face
meanWomen = mean(women_training_pics, 2);
women_training_pics = women_training_pics - repmat(meanWomen, 1, nSmp);

%SVD of both men and women pictures
[um, dm, vm] = svd(men_training_pics, 'econ');
[uw, dw, vw] = svd(women_training_pics, 'econ');

%eigenvalues and eigenvectors of SVD of men pictures
eigenVals_men = diag(dm);
eigenVecs_men = um;

%eigenvalues and eigenvectors of SVD of women pictures
eigenVals_women = diag(dw);
eigenVecs_women = uw;

% The cumulative energy content for the m'th eigenvector is the sum of the energy content across eigenvalues 1:m
energy_m = [];
for i = 1:1296
    energy_m(i) = sum(eigenVals_men(1:i));
end
propEnergy_m = energy_m./energy_m(end);

% Determine the number of principal components required to model 90% of data variance
percentMark_m = min(find(propEnergy_m > 0.9));
percentMark_m = percentMark_m +1;

% Pick those principal components
eigVecs_men = um(:, 1:percentMark_m);

%apply the same process to women
energy_w = [];
for k = 1:1296
    energy_w(k) = sum(eigenVals_women(1:k));
end
propEnergy_w = energy_w ./ energy_w(end);

percentMark_w = min(find(propEnergy_w > 0.9));

eigVecs_women = uw(:, 1:percentMark_w);

%To calculate weights of each gender reduced eigenvectors are multiplied by
%the training pictures of each gender
men_weights = eigVecs_men' * men_training_pics;
women_weights = eigVecs_women' * women_training_pics;

%The means of weights of each gender is taken
mean_men_weights = mean(men_weights, 2);
mean_women_weights = mean(women_weights, 2);
%Training process is complete!

%read all the pictures in the testing/men folder
men_testing_folder = 'testing/men/';
men_testing_struct = dir([men_testing_folder '*.jpg']);

%read all the pictures in the testing/women folder
women_testing_folder = 'testing/women/';
women_testing_struct = dir([women_testing_folder '*.jpg']);

%convert men testing pictures struct to array
men_testing_pics_array = [];
for k = 1:length(men_testing_struct)
    men_test_pic_index = imread([men_testing_folder men_testing_struct(k).name]);
    men_test_pics_index = reshape(men_test_pic_index, 1, []);
    men_testing_pics_array = [men_testing_pics_array, men_test_pics_index]; 
end

%convert women testing pictures struct to array
women_testing_pics_array = [];
for k = 1:length(women_testing_struct)
    women_test_pic_index = imread([women_testing_folder women_testing_struct(k).name]);
    women_test_pics_index = reshape(women_test_pic_index, 1, []);
    women_testing_pics_array = [women_testing_pics_array, women_test_pics_index];
end

%convert women testing pictures array to 1296x200 matrix and convert it
%from uint8 to double and divide by 255 for normalization which is the max
%value
women_testing_pics = reshape(women_testing_pics_array, [1296, 200]);
women_testing_pics = double(women_testing_pics);
women_testing_pics_weights = women_testing_pics ./ 255;

%convert women testing pictures array to 1296x200 matrix and convert it
%from uint8 to double and divide by 255 for normalization which is the max
%value
men_testing_pics = reshape(men_testing_pics_array, [1296, 200]);
men_testing_pics = double(men_testing_pics);
men_testing_pics_weights = men_testing_pics ./ 255;

%create empty arrays for later use
tested_women = [];
tested_men = [];

%we do the classification of women testing pictures
%we multiply each column by eigenvectors/eigenfaces, one by one, and calculate its difference with mean men and women faces
%we then compare the two differences and label that particular vector as whichever has the smaller difference
for i = 1:200
    weightdiff_women = abs((eigVecs_women' * women_testing_pics_weights(:,i)) - mean_women_weights);
    weightdiff_men = abs((eigVecs_women' * women_testing_pics_weights(:,i)) - mean_men_weights);
    if sum(weightdiff_women) < sum(weightdiff_men)
        tested_women = [tested_women, women_testing_pics_weights(:,i)];
    end
    if sum(weightdiff_women) > sum(weightdiff_men)
        tested_men = [tested_men, women_testing_pics_weights(:,i)];
    end
end

%we do the classification of men testing pictures
%we multiply each column by eigenvectors/eigenfaces, one by one, and calculate its difference with mean men and women faces
%we then compare the two differences and label that particular vector as whichever has the smaller difference
for i = 1:200
    weightdiff_women = abs((eigVecs_men' * men_testing_pics_weights(:,i)) - mean_women_weights);
    weightdiff_men = abs((eigVecs_men' * men_testing_pics_weights(:,i)) - mean_men_weights);
    if sum(weightdiff_women) > sum(weightdiff_men)
        tested_men = [tested_men, men_testing_pics_weights(:,i)];
    end
    if sum(weightdiff_women) < sum(weightdiff_men)
        tested_women = [tested_women, men_testing_pics_weights(:,i)];
    end
end

%finally, we display tested_men and tested_women matrices in different figures 
faceW = 36; 
faceH = 36; 
numPerLine = 13; 
ShowLine = 10;
Y_m = zeros(faceH*ShowLine,faceW*numPerLine); 
for i=0:ShowLine-1 
  	for j=0:numPerLine-1 
        a = reshape(tested_men(:,i*numPerLine+j+1),[faceH,faceW]);
    	Y_m(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW) =  a;
  	end 
end 

figure; imagesc(Y_m);colormap(gray); title ('men');


faceW = 36; 
faceH = 36; 
numPerLine = 19; 
ShowLine = 10;
Y_w = zeros(faceH*ShowLine,faceW*numPerLine); 
for i=0:ShowLine-1 
  	for j=0:numPerLine-1 
        a = reshape(tested_women(:,i*numPerLine+j+1),[faceH,faceW]);
    	Y_w(i*faceH+1:(i+1)*faceH,j*faceW+1:(j+1)*faceW) =  a;
  	end 
end 

figure; imagesc(Y_w);colormap(gray); title('women');

