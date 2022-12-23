function [imgf] = CAOL_Fusion( x1, x2,init,flag )
     if size(x1,3)>1
        x1_gray=double(rgb2gray(x1));
        x2_gray=double(rgb2gray(x2));
    else
       x1_gray = x1;
       x2_gray = x2;
     end

    npd = 16;
    fltlmbd = 5;
    [s1_l, s1_h] = lowpass(double(x1_gray), fltlmbd, npd);
    imshow(s1_l,[]);
    imshow(s1_h,[]);
    [s2_l, s2_h] = lowpass(double(x2_gray), fltlmbd, npd);
        imshow(s2_l,[]);
        imshow(s2_h,[]);
    opt = [];
opt.Verbose = 0;
opt.MaxMainIter = 500;
opt.rho = 100*0.01 + 0.5;
opt.RelStopTol = 1e-3;
opt.AuxVarObj = 0;
opt.HighMemSolve = 1;
[X1, optinf1] = cbpdn(init.d, s1_h, 0.01, opt);
[X2, optinf2] = cbpdn(init.d, s2_h, 0.01, opt);

%activity level measure
A1=sum(abs(X1),3);
A2=sum(abs(X2),3);

if flag == 1  
    r=9;  
else
    r=3; 
end

ker=ones(2*r+1,2*r+1)/((2*r+1)*(2*r+1));
AA1=imfilter(A1,ker);
AA2=imfilter(A2,ker);
decisionMap=AA1>AA2;

%base layer fusion
if flag == 1  
    imgf_l=s1_l.*decisionMap+s2_l.*(1-decisionMap);
else
    imgf_l=(s1_l+s2_l)/2;
end
imshow(imgf_l,[]);
%detail layer fusion
[height,width]=size(A1);
X=X1;
for j=1:height
    for i=1:width
        if decisionMap(j,i)==0
            X(j,i,:)=X2(j,i,:);
        end
    end
end
imgf_h = ifft2(sum(bsxfun(@times, fft2(init.d, size(X,1), size(X,2)), fft2(X)),3),'symmetric');
imshow(imgf_h,[]);
imgf=uint8(imgf_l+imgf_h);
    
    
%% 求解稀疏系数via BPEG-M
% disp('BPEG-M for first source test image');
%     [z1, Dz1] = BPEGM_CAOL_2D_TF( s1_h, param.size_kernel, ...
%         param.alpha, param.lambda, param.arcdegree, param.major_type, ...
%         param.max_it, param.tol, verbose_disp, init);
%    disp('BPEG-M for two source test image');
%     [z2, Dz2] = BPEGM_CAOL_2D_TF(s2_h,param.size_kernel, ...
%         param.alpha, param.lambda, param.arcdegree, param.major_type, ...
%         param.max_it, param.tol, verbose_disp, init);
%  %% 
%     A1=sum(abs(z1),3);
%     A2=sum(abs(z2),3);
%     
% if flag == 1  
%     r=9;  
% else
%     r=3; 
% end
% 
% ker=ones(2*r+1,2*r+1)/((2*r+1)*(2*r+1));
% AA1=imfilter(A1,ker);
% AA2=imfilter(A2,ker);
% decisionMap=AA1>AA2;
% 
% %base layer fusion
% if flag == 1  
%     imgf_l=s1_l.*decisionMap+s2_l.*(1-decisionMap);
% else
%     imgf_l=(s1_l+s2_l)/2;
% end
% 
% %detail layer fusion
% [height,width]=size(A1);
% X=z1;
% for j=1:height
%     for i=1:width
%         if decisionMap(j,i)==0
%             X(j,i,:)=z2(j,i,:);
%         end
%     end
% end
% % dnorm = @(x) bsxfun(@rdivide, x, sqrt(sum(sum(x.^2, 1), 2)));
% % d = dnorm(init.d);
% % PSt = @(u) d_to_dpad(u, [size(s1_h,1)  size(s1_h,2) 1], [5  5 25], [5  5]);
% % d_hat = fft2( PSt(d) );
% 
% imgf_h = ifft2(sum(bsxfun(@times, fft2(init.d, size(X,1), size(X,2)), fft2(X)),3),'symmetric');
% % z_hat = reshape( fft2( reshape(X, size(s1_h,1) ,size(s1_h,2) ,[]) ), [ size(s1_h,1)  size(s1_h,2)  25    1] );
% % imgf_h1 = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,1]) .* z_hat, 3), [size(s1_h,1)  size(s1_h,2)]) ));
% % imgf_h2 = imgf_h1(1 + psf_radius1:end - psf_radius1,1 + psf_radius1:end - psf_radius1);
% 
% %% fusion
% disp('fusion');
% if size(x1,3)>1
%     decisionMap = guidedfilter(imgf_h2,decisionMap((1 + psf_radius1):(size(decisionMap,1) - psf_radius1),(1 + psf_radius1):(size(decisionMap,2) - psf_radius1)),8,0.1);
%     decisionMap_rgb=repmat(decisionMap,[1 1 3]);
%     imgf=(double(x1)/255).*decisionMap_rgb+(double(x2)/255).*(1-decisionMap_rgb);
%     imgf=uint8(imgf);
% else 
%     imgf=uint8(imgf_l+imgf_h);
% end
