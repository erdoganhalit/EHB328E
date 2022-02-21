%read the picture
nasacolor=imread('TarantulaNebula.jpg');

%display the picture first as it is
figure; image(nasacolor); title('colored nasa');

nasa_summed=sum(nasacolor,3,'double'); %sum up red+green+blue
m=max(max(nasa_summed)); %find the max value
nasa_normalized=nasa_summed*255/m; %make this be bright white

colormap(gray(256));

%display grayscale photo
figure; imshow(nasa_normalized); title('Grayscale NASA photo');

%apply singular value decomposition
[U, S, V]=svd(nasa_normalized);

%display eigenvalues in a log scale
semilogy(diag(S))

%define different picture matrices with different numbers of eigenvectors
%and eigenvalues
nasa100=U(:,1:100)*S(1:100,1:100)*V(:,1:100)';
nasa50=U(:,1:50)*S(1:50,1:50)*V(:,1:50)';
nasa25=U(:,1:25)*S(1:25,1:25)*V(:,1:25)';

%display each of the pictures defined above
figure; image(nasa100); title('NASA100 photo');

image(nasa50); title('NASA50 photo');

image(nasa25); title('NASA25 photo');

