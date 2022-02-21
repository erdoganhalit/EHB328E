%define empty arrays for later use
neutral = [];
smile = [];

%Nomalize each vector to unit
[nSmp,nFea] = size(fea);
for i = 1:nSmp
     fea(i,:) = fea(i,:) ./ max(1e-12,norm(fea(i,:)));
end

%when fea is displayed with display_faces.m, it is seen that 3rd picture of
%each individual is smiling. We label every (3+11k)th face as smile.
smile = [];
for k = 1:11
    l = (11 .* (k-1)) + 3;
    smile = [smile, l];
end

%The rest of the faces are neutral
neutral = [];
for k = 1:121
    if rem(k,11) == 3
        k = k + 1;
    else
        neutral = [neutral, k];
    end
end

%The transpose is taken to make each face a column
faces = fea';
%sizes of the pictures are defined for plotting
h = 32; w = 32;
%mean face is calculated
meanFace = mean(faces, 2);
%mean face is subtracked from faces
faces = faces - repmat(meanFace, 1, nSmp);

%Singular value decomposition is applied
[u,d,v] = svd(faces, 0);

% Eigenvalues and eigenvectors are calculated
eigVals = diag(d);
eigVecs = u;

%figures are plotted for the mean face and the first 3 eigenvectors/eigenfaces
figure; imshow(reshape(meanFace, h, w)); title('Mean Face');
figure;
subplot(1, 3, 1); imagesc(reshape(u(:, 1), h, w)); colormap(gray); title('First Eigenface');
subplot(1, 3, 2); imagesc(reshape(u(:, 2), h, w)); colormap(gray); title('Second Eigenface'); 
subplot(1, 3, 3); imagesc(reshape(u(:, 3), h, w)); colormap(gray); title('Third Eigenface');

% The cumulative energy content for the m'th eigenvector is the sum of the energy content across eigenvalues 1:m
energy = [];
for i = 1:nSmp
    energy(i) = sum(eigVals(1:i));
end
propEnergy = energy./energy(end);

% Determine the number of principal components required to model 90% of data variance
percentMark = min(find(propEnergy > 0.9));

% Pick those principal components
eigenVecs = u(:, 1:percentMark);

%project each of the neutral and smiling faces onto the corresponding eigenfaces
neutralFaces = faces(:, neutral); 
smileFaces = faces(:, smile);
neutralWeights = eigenVecs' * neutralFaces;
smileWeights = eigenVecs' * smileFaces;

%the means of each weight matrices are taken, giving us a mean neutral and
%smiling weight which we can finally use to classify remaining pictures/vectors 
mean_neutralWeights = mean(neutralWeights, 2);
mean_smileWeights = mean(smileWeights, 2);


%we do the classification
%122 to 165 is the remaining indices and are our test data, since the first 121 is labeled and therefore has become our training data
%we multiply each column by eigenvectors/eigenfaces, one by one, and calculate its difference with mean neutral smiling faces
%we then compare the two differences and label that particular vector as whichever has the smaller difference
for i = 122:165  
    
    weightdiff_smile = abs((eigenVecs' * faces(:,i)) - mean_smileWeights);
    weightdiff_neutral = abs((eigenVecs' * faces(:,i)) - mean_neutralWeights);
    if sum(weightdiff_smile) > sum(weightdiff_neutral)
        neutral = [neutral, i];
    end
    if sum(weightdiff_smile) < sum(weightdiff_neutral)
        smile = [smile, i];
    end
end





    






