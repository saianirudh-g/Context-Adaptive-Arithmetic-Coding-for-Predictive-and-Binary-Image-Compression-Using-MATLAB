%function [A, eigv]=getGFT(x, opt)
function [A, eigv]=getGFT(x, opt)
if nargin==0
  dbg=1;
  figure(31); hold on; grid on; axis([0 1 0 1]);
  for k=1:16
      [u, v]=ginput(1); 
      plot(u,v,'*r'); 
      x(k,:) = [u, v];
  end
  opt.knn = 4; opt.theta = 0.1; opt.d0 = 0.25;
  opt.norm = 1; 
end

[n, kd]=size(x);

% feature distance:
dist = pdist2(x,x); mdist=mean(dist(:));
% heat kernel size: map mean distance affinity to opt.d0
h = -log(opt.d0)/mdist; 
% affinity: 
S1 = exp(-h*dist); 
% sparsify
S1(find(S1<opt.theta)) = 0; 

% knn graph
[~, indx]=min(dist); S2 = zeros(n, n); 

% assignment matrix
for j=1:n 
    [mv, indx]=sort(dist(j,:));
    S2(j, indx(1:opt.knn+1)) =1;
end

% combine knn and feature based affinity
if opt.norm % normalize, freq in [0, 2]
    S = S1;
    dd = sum(S);
    S=S./sqrt(kron(dd, dd'));
else % non-normalized, use knn to normalize
    S=S1.*S2; 
end

% degree 
dd = sum(S);
D=zeros(n, n); 
for k=1:n
    D(k,k) = dd(k);
end

% Laplacian
[A,eigv]=eig(D-S);
f=diag(eigv);


if dbg
    figure(45); 
    subplot(2,3,1); imagesc(dist); title('dist'); 
    subplot(2,3,2); imagesc(S1); title('S_1');
    subplot(2,3,3); imagesc(S2); title('S_2');
    subplot(2,3,4); imagesc(S); title('S');
    subplot(2,3,5); grid on; hold on; stem(f, '.'); title('\lambda');
    subplot(2,3,6); grid on; hold on; plot(A(:,1), '.-r'); plot(A(:,2), '.-b');plot(A(:,3), '.-k'); title('u_k');
   
end

return; 