% Some tests with fast curvelet transform (from L. Demanet's code).
[N1,N2] = size(img);
im=zeros(2^nextpow2(max(N1,N2)),2^nextpow2(max(N1,N2)));
im(1:N1,1:N2)=img;

% Initialisations.
C=fdct_wrapping(im,1);
is_real = 1;
nbscales = length(C) - 1;  % Number of scales.
nbangles_coarse = 16;  % Minimum partition in terms of angles. 
nbangles = fliplr([1, nbangles_coarse .* 2.^(ceil((nbscales-(2:nbscales))/2)), 1]);    % Vector of number 
									 % of orientations per scale.

% Sort the coefficients from the largest to the smallest. 
sC=[];         
for j = 1:(nbscales-1)
  for l = 1:nbangles(j)
     tmp=C{j}{l};   
     sC=[sC tmp(:)'];
  end
end
[S,I]=sort(abs(sC));sC=fliplr(S.*sign(sC(I)));
%semilogy(sC);

% Display the 16 curvelets (backprojection) corresponding to the largest coefficients 
% in magnitude of the FCT of the image.

for k=1:16
 Cnew = cell(1,nbscales+1);
 for j = 1:nbscales
   Cnew{j} = cell(1,nbangles(j));
   for l = 1:nbangles(j)
      Cnew{j}{l}=zeros(size(C{j}{l}));
   end
 end;
 Cnew{end}=C{end};
 for j = 1:(nbscales-1)
   for l = 1:nbangles(j)
      [tx,ty]=find(C{j}{l}==sC(k));
      if ~isempty(tx) 
      	jm=j;lm=l;txm=tx;tym=ty;
	Cnew{j}{l}(tx,ty)=1;
      end
   end
 end
 X=ifdct_wrapping(Cnew,1);
 subplot(4,4,k);
 imagesc(X(1:94,:));axis image;drawnow
 title(sprintf('j=%d l=%d tx=%d ty=%d',jm,lm,txm,tym));
end

% At each scale, display the cuvelets at each orientation and central location.
% The angles start at about pi/4 and then describe the unitary circle in clockwise (anti-trigonometric) sense.
deep    = [nbscales+2 nbscales+2];
b_ridge = MakeRPRidgeletBasis(deep);
raxp = axis;
xgrid=linspace(0,1,N1);
ygrid=linspace(0,1,N2);

for k = 2:nbscales
  for l = 1:nbangles(k)
   Cnew = cell(1,nbscales+1);
   for j = 1:nbscales+1
     Cnew{j} = cell(1,nbangles(j));
     for l2 = 1:nbangles(j)
       Cnew{j}{l2}=zeros(size(C{j}{l2}));
     end
   end

   txy=floor(size(C{k}{l})/2);
   jm=k;lm=l;txm=txy(1);tym=txy(2);
   Cnew{jm}{lm}(txm,tym)=1;
   
  X=ifdct_wrapping(Cnew,1);
  %subplot(ceil(sqrt(nbangles(k))),ceil(sqrt(nbangles(k))),l);
  subplot(121);
  cla
  imagesc(real(X));axis image;rmaxis;drawnow
  subplot(122);
  cla
  imagesc(1-xgrid,ygrid,abs(fftshift(fft2(X))));axis image;drawnow
  colormap(1-hot);
  hold on
  PlotRectoPolarRPPartition(b_ridge,'k',raxp,deep);axis([0.5-2^(-deep(1)+4) 0.5+2^(-deep(1)+4) 0.5-2^(-deep(1)+4) 0.5+2^(-deep(1)+4)]);axis off;
  title(sprintf('j=%d l=%d tx=%d ty=%d',jm,lm,txm,tym));
  pause;
  end
  %clf
end

% Filter out the horizontal and vertical linear structures

%curv=[1  2  3 4 5 1  2  3  4  5;...
%      17 17 8 8 4 48 48 25 25 12];
curv=[1 2 3 1  2  3;...
      8 8 4 25 25 12];
Cnew=C;
for j = 1:length(curv)
  Cnew{curv(1,j)}{curv(2,j)} = zeros(size(C{curv(1,j)}{curv(2,j)}));
  Cnew{curv(1,j)}{curv(2,j)+1} = zeros(size(C{curv(1,j)}{curv(2,j)+1}));
  Cnew{curv(1,j)}{curv(2,j)-1} = zeros(size(C{curv(1,j)}{curv(2,j)-1}));
end
Cnew{nbscales}{1}=zeros(size(C{nbscales}{1})); 
X=ifdct_wrapping(Cnew,1);
imagesc(X);axis image;drawnow

% Separate orthogonally oriented linear structures in two by two quadrants (1-4 and 2-3)
%       -------------
%       |  1  |  2  |
%       -------------
%       |  3  |  4  |
%       -------------

curv=[2*ones(1,16) 3*ones(1,17)  4*ones(1,8);...
      [9:25]     [9:25]        [5:12]];
Cnew=fdct_wrapping(X,1);
for j = 1:length(curv)
  Cnew{curv(1,j)}{curv(2,j)} = zeros(size(C{curv(1,j)}{curv(2,j)}));
end
Cnew{nbscales}{1}=zeros(size(C{nbscales}{1})); 
X2=ifdct_wrapping(Cnew,1);
imagesc(X2);axis image;drawnow



