%%%%%%%%%%%%%%%%%%%%%%%%%%%
% transforms
%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
dct_q = load('../data/dct_q.txt');
im0 = imread('../pics/lena.png'); subplot(2,2,1); imshow(im0); title('lena');
im=rgb2gray(im0); [h, w]=size(im); 
% double array form
imx = double(im(:));

blk_size = 8; 

% break up into blocks
n_h_blk = fix(h/blk_size); n_w_blk = fix(w/blk_size); 
for j=1:n_h_blk
    for k=1:n_w_blk
        h_offs = (j-1)*blk_size; 
        w_offs = (k-1)*blk_size;
        im_blk{j, k}  =  im(h_offs+1:h_offs+blk_size, w_offs+1:w_offs+blk_size);                 
    end
end

% KLT/PCA
if (1)
    % collect training data for PCA
    npt= 4000; im_pca_x = zeros(npt, blk_size*blk_size);
    xy_offs = fix(rand(npt, 2)*(w-blk_size-5)); 
    for k=1:npt
        blk = im(xy_offs(k, 1)+1:xy_offs(k, 1)+blk_size, xy_offs(k, 2)+1:xy_offs(k, 2)+blk_size);
        im_pca_x(k, :) = double(blk(:)');
    end
    [A, s, lat]=princomp(double(im_pca_x));
    %save ../data/pca-8x8-lena-2016.02.08.mat A lat; 
    %save ../data/pca-12x12-lena-2016.02.08.mat A lat; 
    save ../data/pca-16x16-lena-2016.02.08.mat A lat; 
    
else
    if blk_size==8 
        load ../data/pca-8x8-lena-2016.02.08.mat;
    elseif blk_size == 12
        load ../data/pca-12x12-lena-2016.02.08.mat;
    else
        load ../data/pca-16x16-lena-2016.02.08.mat;
    end
end
figure(32); grid on; hold on; stem(lat, '.'); title('eigen values: pca 8x8 ');

% VQ
vq_rate = 7; 
[indx, codebook]=kmeans(im_pca_x*A(:,1:12), 2^vq_rate);
figure(33); plot3(codebook(:,1), codebook(:,2), codebook(:,3), '*r'); grid on; hold on;

% fast VQ with kd-tree 
path(path, '../../ScalableVideoMining/matlab');
% build a kd-tree: each leaf node has an unique prefix code
[kdt.indx, kdt.leafs, kdt.mbox]=buildVisualWordList(im_pca_x*A(:,1:12), vq_rate);
% query
q = im_pca_x(1200:1215,:)*A(:,1:12); 
[kt_node, prefix_code] = searchVisualWordIndex(q, kdt.indx, kdt.leafs);


% reconstruction via PCA/SVD
% 8x8 zigzag
%      1     9    17    25    33    41    49    57
%      2    10    18    26    34    42    50    58
%      3    11    19    27    35    43    51    59
%      4    12    20    28    36    44    52    60
%      5    13    21    29    37    45    53    61
%      6    14    22    30    38    46    54    62
%      7    15    23    31    39    47    55    63
%      8    16    24    32    40    48    56    64
% 12x12 zigzag
% zgz_12 =
% 
%      1    13    25    37    49    61    73    85    97   109   121   133
%      2    14    26    38    50    62    74    86    98   110   122   134
%      3    15    27    39    51    63    75    87    99   111   123   135
%      4    16    28    40    52    64    76    88   100   112   124   136
%      5    17    29    41    53    65    77    89   101   113   125   137
%      6    18    30    42    54    66    78    90   102   114   126   138
%      7    19    31    43    55    67    79    91   103   115   127   139
%      8    20    32    44    56    68    80    92   104   116   128   140
%      9    21    33    45    57    69    81    93   105   117   129   141
%     10    22    34    46    58    70    82    94   106   118   130   142
%     11    23    35    47    59    71    83    95   107   119   131   143
%     12    24    36    48    60    72    84    96   108   120   132   144
%     
% zgz_16 =
%      1    17    33    49    65    81    97   113   129   145   161   177   193
%      2    18    34    50    66    82    98   114   130   146   162   178   194
%      3    19    35    51    67    83    99   115   131   147   163   179   195
%      4    20    36    52    68    84   100   116   132   148   164   180   196
%      5    21    37    53    69    85   101   117   133   149   165   181   197
%      6    22    38    54    70    86   102   118   134   150   166   182   198
%      7    23    39    55    71    87   103   119   135   151   167   183   199
%      8    24    40    56    72    88   104   120   136   152   168   184   200
%      9    25    41    57    73    89   105   121   137   153   169   185   201
%     10    26    42    58    74    90   106   122   138   154   170   186   202
%     11    27    43    59    75    91   107   123   139   155   171   187   203
%     12    28    44    60    76    92   108   124   140   156   172   188   204
%     13    29    45    61    77    93   109   125   141   157   173   189   205
%     14    30    46    62    78    94   110   126   142   158   174   190   206
%     15    31    47    63    79    95   111   127   143   159   175   191   207
%     16    32    48    64    80    96   112   128   144   160   176   192   208
zigzag = reshape([1:blk_size*blk_size], [blk_size, blk_size]);
zgz_8 = [1 2 9 3 10 17 4 11 18 25 5 12 19 26 33 6 13 20 27 34 41];
zgz_12 = [1 2 13 3 14 25 4 15 26 37 5 16 27 38 49];
zgz_16 = [1 2 17 3 18 33 4 19 34 49 5 20 35 50 65];

if blk_size==8
    zigzag_offs = zgz_8;
elseif blk_size == 12
    zigzag_offs = zgz_12;
else
    zigzag_offs = zgz_16;
end


% m-coeff reconstruction
m = 6;
for j=1:n_h_blk
    for k=1:n_w_blk
        % dct
        %dct_coeff = fix(dct2(im_blk{j,k})./dct_q);
        dct_coeff =  dct2(im_blk{j,k});
        dct_mask = zeros(blk_size, blk_size); dct_mask(zigzag_offs(1:m))=1;
        im_blk_dct{j, k} = idct2(dct_coeff.*dct_mask); 
        
        figure(35); 
        subplot(2,2,1); colormap('gray'); imagesc(im_blk{j,k}); title('image');
        subplot(2,2,2); colormap('gray'); imagesc(dct_coeff); title(sprintf('dct coeff %d',m));
        subplot(2,2,3); colormap('gray'); imagesc(fix(dct_coeff./dct_q).*dct_mask); title(sprintf('dct quant %d',m));
        subplot(2,2,4); colormap('gray'); imagesc(idct2(dct_coeff.*dct_mask)); title(sprintf('im recon %d',m));
        
    
        % pca
        pca_coeff = double(im_blk{j,k}(:)')*A; 
        pca_coeff(m+1:end) = 0; 
        im_blk_pca{j,k} = reshape(pca_coeff*inv(A), [blk_size, blk_size]);
        
        % svd
        im_blk_svd{j, k} = svd_approx(double(im_blk{j,k}), m);  
        
        figure(36); 
        subplot(2,2,1); colormap('gray'); imagesc(im_blk{j,k}); title('image');
        subplot(2,2,2); colormap('gray'); imagesc(im_blk_dct{j,k}); title(sprintf('dct %d',m));
        subplot(2,2,3); colormap('gray'); imagesc(im_blk_pca{j,k}); title(sprintf('pca %d',m));
        subplot(2,2,4); colormap('gray'); imagesc(im_blk_svd{j,k}); title(sprintf('svd %d',m'));
        
        figure(37);  grid on; hold on;
        plot(im_blk{j,k}(:), '.-'); 
        plot(im_blk_dct{j,k}(:), '.-r'); 
        plot(im_blk_pca{j,k}(:), '.-m'); 
        plot(im_blk_svd{j,k}(:), '.-k'); 
        legend('im', 'dct', 'pca', 'svd');
        
        
        
    end
end



return;
