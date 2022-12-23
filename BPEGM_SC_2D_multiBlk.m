function [z_res, Dz, psf_radius,psf_s ] = ...
        BPEGM_SC_2D_multiBlk(x, size_kernel, alpha, arcdegree, ...
        Md_type, Mz_type, max_it, tol, verbose, init)     
     
%| BPEGM_CDL_2D_multiBlk:
%| Multi-block ver. Convolutional Dictionary Learning (CDL) via 
%| Block Proximal Extrapolated Gradient method using Majorization and
%| gradient-based restarting scheme (reG-BPEG-M, block multi-convex ver.)
%|
%| [Input]
%| x: training images in sqrt(N) x sqrt(N) x L
%| size_kernel: [psf_s, psf_s, K]
%| alpha: reg. param. for sparsifying regularizer (l1 term)
%| arcdegree: param. for gradient-based restarting, within [90,100] -- 
%|            a larger value leads to more frequent restarting 
%| Md_type: majorization matrix opt. for filter update, 'I','D'
%|          (default: 'D' in Lem. 5.1 of DOI: 10.1109/TIP.2017.2761545)
%| Mz_type: majorization matrix opt. for sparse code update, 'I','D' 
%|          (default: 'D' in Lem. 5.2 of DOI: 10.1109/TIP.2017.2761545)
%| max_it: max number of iterations
%| tol: tolerance value for the relative difference stopping criterion
%| verbose: option to show intermidiate results
%| init: initial values for filters, sparse codes
%|
%| [Output]
%| d_res: learned filters in [psf_s, psf_s, K]
%| z_res: final updates of sparse codes
%| Dz: final synthesized images
%| obj_val: final objective value
%| iterations: records for BPEG-M iterations 
%|
%| [Ref.] DOI: 10.1109/TIP.2017.2761545
%| Copyright 2019-09-10, Il Yong Chun, University of Hawaii
%| alpha ver 2018-03-05, Il Yong Chun, University of Michigan
%|
%| Note: BPEG-M here can be further accelerated by majorization design and 
%| spatial implementation in the CAOL toolbox.


%% Def: Parameters, Variables, and Operators
K = size_kernel(3);     %number of filters
L = size(x,3);          %number of images

%variables for filters
psf_s = size_kernel(1);
center = floor([size_kernel(1), size_kernel(2)]/2) + 1;
psf_radius = floor( psf_s/2 );

%dimensions of padded training images and sparse codes
size_xpad = [size(x,1) + psf_s-1, size(x,2) + psf_s-1, L];
size_z = [size_xpad(1), size_xpad(2), L, K];

%Operator for padding/unpadding to filters
PS = @(u) dpad_to_d(u, center, size_kernel);
PSt = @(u) d_to_dpad(u, size_xpad, size_kernel, center);

%Proximal operator for l1 norm
ProxSparseL1 = @(u, a) sign(u) .* max( abs(u)-a, 0 );

%Mask and padded data
[PBtPB, PBtx] = pad_data(x, center(1), psf_s);

%Adaptive restarting: Cos(theta), theta: angle between two vectors (rad)
omega = cos(pi*arcdegree/180); 

%Objective
objective = @(z, dh) objectiveFunction( z, dh, x, alpha, ...
    psf_radius, center, psf_s, size_z, size_xpad );


%% Initialization
%Initialization: filters
if ~isempty(init.d)
    d = init.d;
else
    %Random initialization
    d = randn(size_kernel);
end
%filter normalization
dnorm = @(x) bsxfun(@rdivide, x, sqrt(sum(sum(x.^2, 1), 2)));
d = dnorm(d);
d_hat = fft2( PSt(d) );
d_p = d;

%Initialization: sparse codes
if ~isempty(init.z)
    z = init.z;
else
    %Random initialization (starts with higher obj. value)
    %z = randn(size_z);
    
    %Fixed initialization
    z = zeros(size_z);
    for l = 1:L
        for k = 1:K
            %circular padding
            if mod(psf_s,2) == 0
                xpad = padarray(x(:,:,l), [center(1)-1, center(2)-1], 'circular', 'both');    %circular pad
                xpad = xpad(1:end-1, 1:end-1, :, :);
            else
                xpad = padarray(x(:,:,l), [center(1)-1, center(2)-1], 'circular', 'both');    %circular pad
            end 
            z(:,:,l,k) = ProxSparseL1(xpad, alpha) / K;
            %zero padding
            %z(:,:,l,k) = ProxSparseL1(PBtx(:,:,l), alpha) / K;
            %z(:,:,l,k) = PBtx(:,:,l) .* ( abs(PBtx(:,:,l)) >= alpha ) / K;
        end
    end
end
z_hat = reshape( fft2( reshape(z, size_z(1), size_z(2), []) ), size_z );
z_p = z;

%ETC
Akdkzk_mat = Ak_for_dkzk_mat( d_hat, z_hat, PBtPB, size_z );
Mdk = zeros(size_kernel);
Mzk = zeros(size_z(1), size_z(2), size_z(4));
tau = 1;            %momentum coeff.
weight = 1-eps;     %delta in Prop. 3.2 of DOI: 10.1109/TIP.2017.2761545

%Save all objective values and timings
iterations.obj_vals_d = [];
iterations.obj_vals_z = [];
iterations.tim_vals = [];
iterations.d = [];

%Initial vals
obj_val = objective(z, d_hat);

%Save all initial vars
iterations.obj_vals(1) = obj_val;
iterations.tim_vals(1) = 0;
iterations.d = cat(4, iterations.d, d );

%Debug progress
% fprintf('Iter %d, Obj %3.3g, Diff %5.5g\n', 0, obj_val, 0)

%Display initializations 
% if verbose == true
%     iterate_fig = figure();
%     filter_fig = figure();
%     display_func(iterate_fig, filter_fig, d, d_hat, z_hat, x, size_xpad, size_z, psf_radius, 0);
% end


%% %%%%%%%%%% Multi-block CDL via reG-BPEG-M %%%%%%%%%%
for i = 1:max_it
    for k = 1:K     
        %% UPDATE: The kth sparse code set, { z_{l,k} : l=1,...,L }
        
        %Compute majorization matrices
        tic; %timing
        if i==1
            Mzk(:,:,k) = maj_for_zk( d_hat(:,:,k), d(:,:,k), Mz_type, center(1), psf_s, size_z );
        else
            Mzk_old = Mzk(:,:,k);
            Mzk(:,:,k) = maj_for_zk( d_hat(:,:,k), d(:,:,k), Mz_type, center(1), psf_s, size_z );
        end
        t_spcd_maj = toc; %timing
        
        %System operators
        A = @(u) A_for_zk( d_hat(:,:,k), PBtPB, L, u );      
        Ah = @(u) Ah_for_zk( conj(d_hat(:,:,k)), L, u );   
        
        tic; %timing
        
        %%%%%%%%%%%%%%%%%%%%% reG-BPEG-M %%%%%%%%%%%%%%%%%%%%%%     
        if i ~= 1
            %Extrapolation with momentum!
            E_zk = weight * min( repmat( (tau_old - 1)/tau, size(Mzk(:,:,k)) ), sqrt( Mzk_old ./ Mzk(:,:,k) ) );
            z_p = z(:,:,:,k) + repmat(E_zk,[1,1,L]) .* (z(:,:,:,k) - z_old(:,:,:,k)); 
        end

        %Proximal mapping
        z_old(:,:,:,k) = z(:,:,:,k);
        Akdkzk_mat(:,:,:,k) = 0;
        PBtx_k = PBtx - sum(Akdkzk_mat, 4);
        if i == 1
            z(:,:,:,k) = ProxSparseL1( z_p(:,:,:,k) - Ah( A(z_p(:,:,:,k)) - PBtx_k ) ./ repmat(Mzk(:,:,k),[1,1,L]), ...
                alpha ./ repmat(Mzk(:,:,k),[1,1,L]) );
        else
            z(:,:,:,k) = ProxSparseL1( z_p - Ah( A(z_p) - PBtx_k ) ./ repmat(Mzk(:,:,k),[1,1,L]), ...
                alpha ./ repmat(Mzk(:,:,k),[1,1,L]) );
        end

        %Gradient-based adaptive restarting
        Mzk_diff = repmat(Mzk(:,:,k),[1,1,L]) .* (z(:,:,:,k)-z_old(:,:,:,k));
        if i == 1
            zp_z_diff = z_p(:,:,:,k)-z(:,:,:,k);
        else
            zp_z_diff = z_p-z(:,:,:,k);
        end
        if dot( zp_z_diff(:), Mzk_diff(:) ) / ( norm(zp_z_diff(:)) * norm(Mzk_diff(:)) ) > omega
            z(:,:,:,k) = ProxSparseL1( z_old(:,:,:,k) - Ah( A(z_old(:,:,:,k)) - PBtx_k ) ./ repmat(Mzk(:,:,k),[1,1,L]), ...
                   alpha ./ repmat(Mzk(:,:,k),[1,1,L]) );
%             disp('Restarted!');
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [Akdkzk_mat(:,:,:,k), z_hat(:,:,:,k)] = A( z(:,:,:,k) );
        
        %timing
%         t_spcd_bpegm = toc;
%         t_spcd = t_spcd + t_spcd_maj + t_spcd_bpegm;
           
    end
   
    
    %% UPDATE: Momentum coeff.
    tau_old = tau;
    tau = ( 1 + sqrt(1 + 4*tau^2) ) / 2;
    
    
    %% EVALUATION
    %Debug progress
    obj_val = objective(z, d_hat);            
% %     d_relErr = norm(d(:)-d_old(:),2)/norm(d(:),2);
    z_relErr = norm(z(:)-z_old(:),2)/norm(z(:),2);
%     fprintf('Iter %d, Obj %3.3g, Diff Z %5.5g \n', i, obj_val, ...
%              z_relErr);
%          obj_val1 = size(max_it/10);
%          z_relErr1 = [1 max_it/10];
%          if rem(i,10) ==0
%              for ii= 1:(max_it/10)
%             obj_val1(ii)= obj_val;
%             z_relErr1(ii) =z_relErr;
%              end
%          end
    %fprintf('z_max = %g, ', max(z(:))); fprintf('z_min = %g, ', min(z(:)));
    %fprintf('sparsity perc. = %g\n', nnz(z)/numel(z) * 100);  
       
    %Record current iteration
%     iterations.obj_vals(i + 1) = obj_val;
%     iterations.tim_vals(i + 1) = iterations.tim_vals(i) + t_kernel + t_spcd;
%     if mod(i,100) == 0  %save filters every 100 iteration
%         iterations.d = cat(4, iterations.d, d );
%     end
%     
    %Display intermediate results 
%     if verbose == true        
%         if mod(i,25) == 0
%             display_func(iterate_fig, filter_fig, d, d_hat, z_hat, x, size_xpad, size_z, psf_radius, i);
%         end
%     end

    %Termination
    if (z_relErr< tol) && (i > 1)
        disp('relErr reached');      
        break;
    end
    
end

%Final estimate
z_res = z;
Dz = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,L]) .* permute(z_hat,[1,2,4,3]), 3), size_xpad) ));
    
return;





%%%%%%%%%%%%%%%%%%%% Def: Padding Operators %%%%%%%%%%%%%%%%%%%%

function d = dpad_to_d(dpad, center, size_kernel)

%unpad filters (for multiple filters)
d = circshift(dpad, -[1-center(1), 1-center(2), 0]);
d = d( 1:size_kernel(1), 1:size_kernel(2), : );

return;


function dpad = d_to_dpad(d, size_xpad, size_kernel, center)

%pad filters (for multiple filters)
dpad = padarray( d, [size_xpad(1)-size_kernel(1), size_xpad(2)-size_kernel(2), 0], 0, 'post' );
dpad = circshift(dpad, [1-center(1), 1-center(2), 0]);

return;


function [MtM, Mtx] = pad_data(x, center, psf_s)

if mod(psf_s,2) == 0
    MtM = padarray(ones(size(x)), [center-1, center-1, 0], 0, 'both');    %mask
    MtM = MtM(1:end-1, 1:end-1, :);
    Mtx = padarray(x, [center-1, center-1, 0], 0, 'both');    %padded x
    Mtx = Mtx(1:end-1, 1:end-1, :);
else
    MtM = padarray(ones(size(x)), [center-1, center-1, 0], 0, 'both');    %mask
    Mtx = padarray(x, [center-1, center-1, 0], 0, 'both');    %padded x
end
        
return;




%%%%%%%%%%%%%%%%%%%% Def: System Operators %%%%%%%%%%%%%%%%%%%%

function Axi = A_for_dk( zk_hat, mask, L, PSt, xi )

Axi = real( ifft2( repmat( fft2( PSt( xi ) ), [1,1,L] ) .* zk_hat ) );
Axi(~mask) = 0;  %consider PB'PB

return;


function Ahxi = Ah_for_dk( zkH_hat, PS, xi )

Ahxi = PS( real( ifft2( sum( fft2( xi ) .* zkH_hat, 3 ) ) ) );

return;


function Axi = Ak_for_dkzk_mat( dtld, z_hat, mask, size_z )

%Size
sy = size_z(1); sx = size_z(2); L = size_z(3); K = size_z(4);

Axi = zeros(sy,sx,L,K);
for k = 1:K
    zk_hat = z_hat(:,:,:,k);
    Akxik = real( ifft2( repmat( dtld(:,:,k), [1,1,L] ) .* zk_hat ) );
    Akxik(~mask) = 0;    
    Axi(:,:,:,k) = Akxik;
end

return;


function [Axi, xi_hat] = A_for_zk( dk_hat, mask, L, xi )

xi_hat = fft2(xi);
Axi = real( ifft2( xi_hat .* repmat(dk_hat, [1,1,L]) ) );
Axi(~mask) = 0;  %kron(I_L, PB'*PB) * A

return;


function Ahxi = Ah_for_zk( dH_hat, L, xi )


Ahxi = real( ifft2( fft2(xi) .* repmat(dH_hat, [1,1,L]) ) );

return;




%%%%%%%%%%%%%%%%%%%% Design: Majorization Matrices %%%%%%%%%%%%%%%%%%%%

function Mdk = maj_for_dk( zk_hat, Md_type, size_kernel, PS, PSt )  
% Compute majorizer for filter updates

dy = size_kernel(1); dx = size_kernel(2);   %size

if strcmp( Md_type, 'I')
    %Scaled identity majorization matrix
    
    Mdk_filt = sum(abs(zk_hat).^2, 3);
    Mdk_filt_max = max(Mdk_filt(:));
    Mdk = Mdk_filt_max * PS( PSt( ones(dy, dx) ) ); 
   
elseif strcmp( Md_type, 'D')
    %Diagonal majorization matrix in
    %Lem. 5.1 of DOI: 10.1109/TIP.2017.2761545
    
    Mdk_filt = abs( ifft2( sum(abs(zk_hat).^2, 3) ) );
    Mdk_freq = fft2( Mdk_filt );
    PSt1_freq = fft2( PSt( ones(dy, dx) ) );
    Mdk = PS( real( ifft2( Mdk_freq .* PSt1_freq ) ) );
    
else   
   error('Err: Choose the appropriate  majorization matrix type.'); 
end


return;


function Mzk = maj_for_zk( dk_hat, dk, Mz_type, center, psf_s, size_z )
% Compute majorizer for sparse code updates

sy = size_z(1); sx = size_z(2); K = size_z(3); L = size_z(4);   %size

if strcmp( Mz_type, 'I')
    %Scaled identity majorization matrix
    
    Mzk_filt = abs(dk_hat).^2;
    Mzk_filt_max = max(Mzk_filt(:));
    Mzk = Mzk_filt_max * ones(sy, sx); 

elseif strcmp( Mz_type, 'D')
    %Diagonal majorization matrix in
    %Lem. 5.2 of DOI: 10.1109/TIP.2017.2761545
    
    Psik_one = conv2(ones(sy,sx), abs(dk), 'valid');
    if mod(psf_s,2) == 0
        PBT_Psik_one = padarray(Psik_one, [center-1, center-1, 0], 'both');    %padded b
        PBT_Psik_one = PBT_Psik_one(1:end-1, 1:end-1, :);
    else
        PBT_Psik_one = padarray(Psik_one, [center-1, center-1, 0], 'both');    %padded b
    end
    Mzk = conv2(PBT_Psik_one, rot90(abs(dk)), 'same');
    
else  
    error('Err: Choose the appropriate  majorization matrix option.'); 
end

return;




%%%%%%%%%%%%%%%%%%%% Def: Proximal Operators %%%%%%%%%%%%%%%%%%%%

function u = KernelConstraintProj( u, Md, Md_type )
        
u_norm = norm(u(:),2)^2;

if u_norm > 1
    if strcmp( Md_type, 'I')
        %Projection onto unit sphere constraint
        u = u ./ sqrt(u_norm);
            
    elseif strcmp( Md_type, 'D') || strcmp( Md_type, 'A')
        %Solve (1/2) u' Md u - y' Md u, s.t. u'u <= 1
        Mdu = Md .* u;
        Mdu2 = Mdu.^2;

        %Parameters for Newton's method
        rad = 1;      %unit sphere
        lambda0 = 0;    %initial value
        tol = 1e-6;     %tolerance
        max_iter = 10;  %max number of iterations

        %Optimization with Newton's method
        lambda_opt = Newton( @(x) f_lambda(Mdu2, Md, x), @(x) df_lambda(Mdu2, Md, x), rad, lambda0, tol, max_iter );
        
        u = Mdu ./ ( Md + lambda_opt );
        
    else
        error('Err: Choose the appropriate majorizer option.');
    end    
end

return;


function fval = f_lambda(Mdu2, Md, lambda)

    fval = sum( Mdu2(:) ./ (Md(:) + lambda).^2 );

return;


function dfval = df_lambda(Mdu2, Md, lambda)

    dfval = -2 * sum( Mdu2(:) ./ (Md(:) + lambda).^3 );

return;





%%%%%%%%%%%%%%%%%%%% MISC %%%%%%%%%%%%%%%%%%%%

function f_val = objectiveFunction( z, d_hat, x, alpha, psf_radius, center, psf_s, size_z, size_xpad )
    
    sy = size_z(1); sx = size_z(2); L = size_z(3); K = size_z(4);   %size

    %Sparse representation error + sparsity
    ztld = reshape( fft2(reshape(z,size_z(1),size_z(2),[])), size_z );
    Dz = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,L]) .* permute(ztld,[1,2,4,3]), 3), size_xpad) ));
    
    if mod(psf_s,2) == 0
        f_z = 0.5 * norm( reshape( Dz(center(1):end - (psf_s-center(1)), ...
            center(2):end - (psf_s-center(2)), :) - x, [], 1) , 2 )^2;
    else        
        f_z = 0.5 * norm( reshape( Dz(1 + psf_radius:end - psf_radius,1 + psf_radius:end - psf_radius,:) - x, [], 1) , 2 )^2;
    end    
    g_z = alpha * sum( abs( z(:) ), 1 );
    
    %Function val
    f_val = f_z + g_z;
    
return;


function [] = display_func(iterate_fig, filter_fig, d, d_hat, z_hat, x, size_xpad, size_z, psf_radius, iter)

    sy = size_z(1); sx = size_z(2); L = size_z(3); K = size_z(4);   %size

    figure(iterate_fig);
    Dz = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,L]) .* permute(z_hat,[1,2,4,3]), 3), size_xpad) ));
    Dz = Dz(1 + psf_radius:end - psf_radius,1 + psf_radius:end - psf_radius,:);

    subplot(3,2,1), imagesc(x(:,:,1));  axis image, colormap gray; colorbar; title('Orig');
    subplot(3,2,2), imagesc(Dz(:,:,1)); axis image, colormap gray; colorbar; title(sprintf('Local iterate %d',iter));
    subplot(3,2,3), imagesc(x(:,:,3));  axis image, colormap gray; colorbar;
    subplot(3,2,4), imagesc(Dz(:,:,3)); axis image, colormap gray; colorbar;
    subplot(3,2,5), imagesc(x(:,:,7));  axis image, colormap gray; colorbar;
    subplot(3,2,6), imagesc(Dz(:,:,7)); axis image, colormap gray; colorbar;
    drawnow;

    figure(filter_fig);
    sqr_k = ceil(sqrt(size(d,3)));
    pd = 1;
    d_disp = zeros( sqr_k * [psf_radius*2+1 + pd, psf_radius*2+1 + pd] + [pd, pd]);
    for j = 0:size(d,3) - 1
        d_curr = d(:,:,j+1);
        d_disp( floor(j/sqr_k) * (size(d_curr,1) + pd) + pd + (1:size(d_curr,1)) , mod(j,sqr_k) * (size(d_curr,2) + pd) + pd + (1:size(d_curr,2)) ) = d_curr;
    end
    imagesc(d_disp), colormap gray, axis image, colorbar, title(sprintf('Local filter iterate %d',iter));
    drawnow;
        
return;