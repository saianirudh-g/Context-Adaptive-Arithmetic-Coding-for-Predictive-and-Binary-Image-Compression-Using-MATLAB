%% ECE5578 Homework 2 - Arithmetic Coding
% Complete self-contained MATLAB solution for:
% 1(a) Exp-Golomb + adaptive binary arithmetic coding
% 1(b) q = 1,2,4,8, reconstruction and PSNR
% 1(c) bonus: previous-bit 0/1 context model for Exp-Golomb stream
% 2    8-context arithmetic coding of the binary image
%
% Put lena_color.png in the same folder as this script.
% The predictor is:
%   xhat(i,j) = (x(i-1,j-1)+x(i-1,j)+x(i,j-1))/3
%
% The Exp-Golomb codebook follows the homework:
% 0            -> 0
% +/-1         -> 10 + 1 level bit
% +/-2, +/-3   -> 110 + 2 level bits
% +/-4...+/-7  -> 1110 + 3 level bits
% etc.

clear; clc; close all;

%% Read Lena and convert to grayscale
fname = 'lena_color.png';
im0 = imread(fname);
if size(im0,3) == 3
    im = rgb2gray(im0);
else
    im = im0;
end
im = uint8(im);
[H,W] = size(im);
Npix = H*W;

fprintf('Image size: %d x %d\n',H,W);

%% Predictor display
figure('Name','Original and Binary Image');
subplot(1,2,1);
imshow(im);
title('Grayscale Lena');

%% 1(a) and 1(b): q = 1,2,4,8
qList = [1 2 4 8];

results = zeros(numel(qList),8);
% columns:
% q, rawExpGolombBits, ACbits, residual_bpp, fullImage_bpp,
% fullImage_bpp_with_boundary, PSNR, bonus_context_bpp

for qi = 1:numel(qList)
    q = qList(qi);

    % -------------------------------------------------------------
    % Encoder residual exactly following the homework predictor
    % -------------------------------------------------------------
    resQ = makeResidual(im,q);

    % residual to code is (H-1)x(W-1)
    codedResidual = resQ(2:end,2:end);

    % -------------------------------------------------------------
    % Exp-Golomb binarization
    % -------------------------------------------------------------
    egBits = expGolombBinarize(codedResidual);

    % -------------------------------------------------------------
    % Adaptive binary arithmetic coding: one global 0/1 model
    % -------------------------------------------------------------
    ctx0 = ones(size(egBits));        % all symbols use context #1
    acBits = arithmeticEncodeAdaptive(egBits,ctx0,1);

    rawBits = numel(egBits);
    nAC = numel(acBits);

    residualBpp = nAC / ((H-1)*(W-1));
    fullImageBpp = nAC / Npix;

    % First row + first column are transmitted directly (8 bits/sample)
    boundarySamples = H + W - 1;
    fullBppBoundary = (nAC + 8*boundarySamples) / Npix;

    % -------------------------------------------------------------
    % Decoder-side predictive reconstruction
    % -------------------------------------------------------------
    rec = reconstructFromResidual(im,resQ,q);
    mse = mean((double(im(:))-rec(:)).^2);
    if mse == 0
        p = Inf;
    else
        p = 10*log10(255^2/mse);
    end

    % -------------------------------------------------------------
    % Bonus 1(c): 0/1 context model
    % Here the immediately previous Exp-Golomb bit selects one of
    % two adaptive probability models.
    % context 1 = previous bit 0
    % context 2 = previous bit 1
    % -------------------------------------------------------------
    bonusCtx = ones(size(egBits));
    if numel(egBits) > 1
        bonusCtx(2:end) = double(egBits(1:end-1)) + 1;
    end
    acBitsBonus = arithmeticEncodeAdaptive(egBits,bonusCtx,2);
    bonusBpp = numel(acBitsBonus)/Npix;

    results(qi,:) = [q rawBits nAC residualBpp fullImageBpp ...
                     fullBppBoundary p bonusBpp];

    fprintf('\nq = %d\n',q);
    fprintf('  Exp-Golomb input bits                  = %d\n',rawBits);
    fprintf('  Arithmetic-coded bits                  = %d\n',nAC);
    fprintf('  bpp over residual (%dx%d)             = %.6f\n',H-1,W-1,residualBpp);
    fprintf('  bpp over full image                    = %.6f\n',fullImageBpp);
    fprintf('  bpp incl. direct first row/column      = %.6f\n',fullBppBoundary);
    fprintf('  PSNR                                   = %.4f dB\n',p);
    fprintf('  Bonus previous-bit-context bpp         = %.6f\n',bonusBpp);

    % Save actual bit stream if desired
    save(sprintf('HW2_q%d_bitstream.mat',q),'acBits','egBits','resQ','q');

    figure('Name',sprintf('q=%d',q));
    subplot(1,3,1);
    imagesc(resQ); axis image; colormap gray;
    title(sprintf('Quantized residual q=%d',q));
    subplot(1,3,2);
    imshow(uint8(min(max(round(rec),0),255)));
    title(sprintf('Reconstruction q=%d',q));
    subplot(1,3,3);
    imagesc(double(im)-rec); axis image; colormap gray;
    title(sprintf('Error, PSNR=%.2f dB',p));
end

%% Print result table
T = array2table(results, ...
    'VariableNames',{'q','ExpGolombBits','ArithmeticBits','ResidualBpp', ...
    'FullImageBpp','FullBppWithBoundary','PSNR_dB','BonusContextBpp'});
disp(T);

%% 2. Binary image and 8-context adaptive binary arithmetic coding
% Homework threshold: pixel >= 128 is 1, otherwise 0
bim = uint8(im >= 128);

subplot(1,2,2);
imshow(bim);
title('Binary Lena, threshold = 128');

[symbols,contexts] = binaryImageContexts(bim);

% 8 adaptive models, one for each (a,b,c)
binaryACBits = arithmeticEncodeAdaptive(symbols,contexts,8);
binaryBpp = numel(binaryACBits)/Npix;

% For comparison: no context, one global adaptive model
globalCtx = ones(size(symbols));
binaryGlobalBits = arithmeticEncodeAdaptive(symbols,globalCtx,1);
binaryGlobalBpp = numel(binaryGlobalBits)/Npix;

fprintf('\nQuestion 2: binary image\n');
fprintf('  8-context arithmetic-coded bits = %d\n',numel(binaryACBits));
fprintf('  8-context average rate          = %.6f bpp\n',binaryBpp);
fprintf('  Global-model comparison         = %.6f bpp\n',binaryGlobalBpp);

save('HW2_binary_8context_bitstream.mat','binaryACBits','symbols','contexts','bim');

%% ================================================================
% Local functions
% ================================================================

function resQ = makeResidual(im,q)
% Predictor uses upper-left, upper, and left samples.
    x = double(im);
    [H,W] = size(x);

    pred = zeros(H,W);
    pred(2:end,2:end) = ...
        (x(1:end-1,1:end-1) + ...
         x(1:end-1,2:end)   + ...
         x(2:end,1:end-1))/3;

    res = x - pred;

    % Homework states first row and first column are sent directly,
    % therefore they are not part of residual coding.
    res(1,:) = 0;
    res(:,1) = 0;

    % MATLAB fix() truncates toward zero, matching the supplied homework code
    resQ = fix(res/q);
end

function bits = expGolombBinarize(residual)
% Implements the exact category structure shown in the assignment.
% Scan is row-major for a conventional raster scan.

    [H,W] = size(residual);

    % Conservative allocation; enlarged automatically if required.
    bits = zeros(1,8*numel(residual)+64,'uint8');
    p = 0;

    for i = 1:H
        for j = 1:W
            v = residual(i,j);

            if v == 0
                p = p+1;
                bits(p) = 0;
                continue;
            end

            av = abs(v);
            k = floor(log2(av)) + 1;

            % category prefix = k ones followed by zero
            need = 2*k+1;
            if p+need > numel(bits)
                bits = [bits zeros(1,max(numel(bits),need+1024),'uint8')]; %#ok<AGROW>
            end

            bits(p+1:p+k) = 1;
            p = p+k+1;       % last added location is category terminating zero
            bits(p) = 0;

            % Fixed-level code mapping from the homework table:
            % k=1: -1 -> 0, +1 -> 1
            % k=2: -3,-2,+2,+3 -> 00,01,10,11
            if v < 0
                idx = v + (2^k - 1);
            else
                idx = v;
            end

            for b = k-1:-1:0
                p = p+1;
                bits(p) = uint8(bitget(uint32(idx),b+1));
            end
        end
    end

    bits = bits(1:p);
end

function out = arithmeticEncodeAdaptive(symbols,contexts,nCtx)
% Binary arithmetic coder with:
%   - 32-bit coding interval
%   - Laplace initialization [1,1]
%   - adaptive 0/1 counts
%   - E1/E2/E3 renormalization
%   - count rescaling
%
% symbols  : uint8 row vector of 0/1 values
% contexts : integer row vector in 1..nCtx

    symbols = uint8(symbols(:)');
    contexts = double(contexts(:)');

    TOP     = 2^32 - 1;
    HALF    = 2^31;
    QUARTER = 2^30;
    THREEQ  = 3*QUARTER;

    low = 0;
    high = TOP;
    pending = 0;

    count0 = ones(1,nCtx);
    count1 = ones(1,nCtx);

    % Output normally cannot exceed this by much. Grow if needed.
    out = zeros(1,max(1024,3*numel(symbols)+64),'uint8');
    op = 0;

    for n = 1:numel(symbols)
        s = double(symbols(n));
        c = contexts(n);

        c0 = count0(c);
        c1 = count1(c);
        total = c0+c1;

        range = high-low+1;
        split = low + floor(range*c0/total) - 1;

        if s == 0
            high = split;
        else
            low = split+1;
        end

        % E1/E2/E3 scaling
        while true
            if high < HALF
                [out,op] = emitBit(out,op,0,pending);
                pending = 0;

            elseif low >= HALF
                [out,op] = emitBit(out,op,1,pending);
                pending = 0;
                low = low-HALF;
                high = high-HALF;

            elseif low >= QUARTER && high < THREEQ
                pending = pending+1;
                low = low-QUARTER;
                high = high-QUARTER;

            else
                break;
            end

            low = 2*low;
            high = 2*high+1;
        end

        % Update selected probability model
        if s == 0
            count0(c) = count0(c)+1;
        else
            count1(c) = count1(c)+1;
        end

        % Keep totals bounded
        if count0(c)+count1(c) >= 16384
            count0(c) = max(1,ceil(count0(c)/2));
            count1(c) = max(1,ceil(count1(c)/2));
        end
    end

    % Finalization
    pending = pending+1;
    if low < QUARTER
        [out,op] = emitBit(out,op,0,pending);
    else
        [out,op] = emitBit(out,op,1,pending);
    end

    out = out(1:op);
end

function [out,op] = emitBit(out,op,b,pending)
% Emit b and then the pending opposite bits.
    required = 1+pending;
    if op+required > numel(out)
        out = [out zeros(1,max(numel(out),required+1024),'uint8')]; %#ok<AGROW>
    end

    op = op+1;
    out(op) = uint8(b);

    if pending > 0
        out(op+1:op+pending) = uint8(1-b);
        op = op+pending;
    end
end

function rec = reconstructFromResidual(im,resQ,q)
% Decoder reconstruction from the directly transmitted first row/column
% and the quantized residuals.
    x = double(im);
    [H,W] = size(x);

    rec = zeros(H,W);
    rec(1,:) = x(1,:);
    rec(:,1) = x(:,1);

    for i = 2:H
        for j = 2:W
            pred = (rec(i-1,j-1)+rec(i-1,j)+rec(i,j-1))/3;
            rec(i,j) = pred + q*resQ(i,j);

            % Legal 8-bit image range
            rec(i,j) = min(255,max(0,rec(i,j)));
        end
    end
end

function [symbols,contexts] = binaryImageContexts(bim)
% Context in homework:
%       a b 0
%       c x 0
%       0 0 0
%
% a = upper-left, b = upper, c = left.
% Context number = binary abc + 1, hence 1..8.

    [H,W] = size(bim);
    N = H*W;

    symbols = zeros(1,N,'uint8');
    contexts = ones(1,N);

    p = 0;
    for i = 1:H
        for j = 1:W
            p = p+1;

            if i > 1 && j > 1
                a = double(bim(i-1,j-1));
            else
                a = 0;
            end

            if i > 1
                b = double(bim(i-1,j));
            else
                b = 0;
            end

            if j > 1
                c = double(bim(i,j-1));
            else
                c = 0;
            end

            contexts(p) = 4*a + 2*b + c + 1;
            symbols(p) = bim(i,j);
        end
    end
end
