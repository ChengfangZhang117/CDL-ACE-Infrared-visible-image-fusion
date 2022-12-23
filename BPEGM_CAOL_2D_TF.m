function [z_res, A_dkxl_res] = ...
        BPEGM_CAOL_2D_TF(x, size_kernel, alpha, lambda, arcdegree, ...
        M_type, max_it, tol, verbose, init_d)
    
%| BPEGM_CAOL_2D_TF:
%| Convolutional Analysis Operator Learning (CAOL) w/ tight-frame (TF) cond. 
%| via Block Proximal Extrapolated Gradient method using Majorization and
%| gradient-based restarting scheme (reBPEG-M, block multi-nonconvex ver.)
%|
%| [Input]
%| x: training images in sqrt(N) x sqrt(N) x L
%| size_kernel: [psf_s, psf_s, K]
%| alpha: reg. param. for sparsifying regularizer (l0 term)
%| lambda: scaling param. for majorization matrix, larger than 1 -- 
%|         a larger value leads to looser majorization
%| arcdegree: param. for gradient-based restarting, within [90,100] -- 
%|            a larger value leads to more frequent restarting 
%| M_type: majorization matrix opt. for filter update, 'H','I'
%|         (default: 'H' in Prop. 5.1 of DOI: 10.1109/TIP.2019.2937734)
%| max_it: max number of iterations
%| tol: tolerance value for the relative difference stopping criterion
%| verbose: option to show intermidiate results
%| init: initial values for filters, sparse codes
%|
%| [Output]
%| d_res: learned filters in [psf_s, psf_s, K]
%| z_res: final updates of sparse codes
%| A_dkxl_res: final filtered images via learned sparsifying filters
%| obj_val: final objective value
%| iterations: records for BPEG-M iterations 
%|
%| [Ref.] DOI: 10.1109/TIP.2019.2937734
%| Copyright 2019-09-10, Il Yong Chun, University of Hawaii
%| alpha ver 2019-01-26, Il Yong Chun, University of Michigan


%% Def: Parameters, Variables, and Operators
if size_kernel(1)*size_kernel(2) > size_kernel(3)
    error('The number of filters must be equal or larger than size of the filters.');
end
K = size_kernel(3);     %number of filters
L = size(x,3);          %number of training images

%variables for filters
psf_radius = floor( [size_kernel(1)/2, size_kernel(2)/2] );

%dimensions of training images and sparse codes
size_x = [size(x,1), size(x,2), L];
size_z = [size(x,1), size(x,2), K, L];

%Coordinates on filters (only for odd-sized filters)
[kern_xgrid, kern_ygrid] = meshgrid(-psf_radius(1) : psf_radius(2), ...
                            -psf_radius(2) : psf_radius(2));

%Pad training images (default: circular boundary condition)
[~, xpad] = PadFunc(x, psf_radius);

%Proximal operators for l0 "norm"
ProxSparseL0 = @(u, theta) u .* ( abs(u) >= theta );

%Convolutional operators
A_k = @(u) A_for_dk( xpad, u, size_x );
Ah_k = @(u) Ah_for_dk( xpad, u, size_x, size_kernel );
A_kl = @(u) A_for_dk_xl( xpad, u, size_z );

%Majorization matrix design for filter updating problems
disp('Pre-computing majorizastion matrices...');
if strcmp(M_type, 'I')
    %Scaled identity majorization matrix in
    %Lem. 5.2 of DOI: 10.1109/TIP.2019.2937734
    
    Md = majorMat_Ak_diag( xpad, size_x, size_kernel );

elseif strcmp(M_type, 'H')
    %Exact Hessian matrix in
    %Prop. 5.1 of DOI: 10.1109/TIP.2019.2937734
    
    Md = majorMat_Ak_full( xpad, size_x, size_kernel, L, psf_radius, kern_xgrid, kern_ygrid );

else
    error('Choose an appropriate majorizer type.');
end
disp('Majorizer pre-computed...!');

%scaled majorization matrix
Ad = lambda * Md;

%adaptive restarting: Cos(theta), theta: angle between two vectors (rad)
omega = cos(pi*arcdegree/180);  

%Objective
objective = @(z, A_dkxl) objectiveFunction( z, A_dkxl, alpha );


%% Initialization
%Initialization: filters
if ~isempty(init_d)
    d = init_d.d;
else
    %Random initialization
    d = randn(size_kernel);
end
%set the first filter as a DC filter
% d(:,:,1) = 1;
%filter normalization
for k = 1:K
    d(:,:,k) = d(:,:,k) ./ (sqrt(size_kernel(1)*size_kernel(2))*norm(d(:,:,k),'fro'));
end    
d_p = d;        

%Initialization: sparse codes
A_dkxl = A_kl(d);        
z = ProxSparseL0( A_dkxl, sqrt(2*alpha) );

%ETC
tau = 1;            %momentum coeff.            
weight = 1-eps;     %delta in (7) of 10.1109/TIP.2019.2937734

%Save all objective values and timings
iterations.obj_vals = [];
iterations.tim_vals = [];
iterations.it_vals = [];

%Initial vals
obj_val = objective(z, A_dkxl);
    
%Save all initial vars
iterations.obj_vals(1) = obj_val;
iterations.tim_vals(1) = 0;
iterations.it_vals = cat(4, iterations.it_vals, d);

%Debug progress
fprintf('Iter %d, Obj %3.3g, Diff %5.5g\n', 0, obj_val, 0)

%Display initializations 
% if verbose == 1
%     iterate_fig = figure();
%     filter_fig = figure();
%     display_func(iterate_fig, filter_fig, d, z, z, x, psf_radius(1), 0);
% end
   
  
%% %%%%%%%%%% Two-block CAOL via reBPEG-M %%%%%%%%%%
for i = 1 : max_it
            
    %% UPDATE: All filters, { d_k : k=1,...,K }
    
    tic; %timing
    %%%%%%%%%%%%%%%%%%%%% reG-BPEG-M %%%%%%%%%%%%%%%%%%%%%%  
    if i ~= 1
        %Extrapolation with momentum!
        w_d = min( (tau_old - 1)/tau, weight*0.5*(lambda-1)/(lambda+1) );
        d_p = d + w_d .* (d - d_old);
    end
   
    %Proximal mapping
    d_old = d;
    Adnu = zeros(size_kernel);
    for k = 1 : K
        if strcmp(M_type, 'I')
            Adnu(:,:,k) = Ad .* d_p(:,:,k) - Ah_k( A_k(d_p(:,:,k)) ...
                - reshape(z(:,:,k,:), [size_z(1),size_z(2),size_z(4)]) ); 
        elseif strcmp(M_type, 'H')
            AhA_dp = Md * reshape(d_p(:,:,k),[],1);
            Addp_m_AhAdp = (lambda-1) * AhA_dp;
            Adnu(:,:,k) = reshape(Addp_m_AhAdp, [size_kernel(1), size_kernel(2)]) + ...
                Ah_k( reshape(z(:,:,k,:), [size_z(1),size_z(2),size_z(4)]) );  
        else
            error('Choose an appropriate majorizer type.');
        end
    end
    d = ProxFilterTightFrame( Adnu, size_kernel );
    
    %Gradient-based adaptive restarting
    if strcmp(M_type, 'I')
        Ad_diff = repmat(Ad,[1,1,K]) .* (d-d_old);
    elseif strcmp(M_type, 'H')
        Ad_diff = reshape( Ad * reshape(d-d_old, [size_kernel(1)*size_kernel(2), size_kernel(3)]), size_kernel );
    else
        error('Choose an appropriate majorizer type.');
    end
    if dot( d_p(:)-d(:), Ad_diff(:) ) / ( norm(d_p(:)-d(:)) * norm(Ad_diff(:)) ) > omega
        d_p = d;
        Adnu = zeros(size_kernel);
        for k = 1 : K
            if strcmp(M_type, 'I')
                Adnu(:,:,k) = Ad .* d_p(:,:,k) - Ah_k( A_k(d_p(:,:,k)) ...
                    - reshape(z(:,:,k,:), [size_z(1),size_z(2),size_z(4)]) );
            elseif strcmp(M_type, 'H')
                AhA_dp = Md * reshape(d_p(:,:,k),[],1);
                Addp_m_AhAdp = (lambda-1) * AhA_dp;
                Adnu(:,:,k) = reshape(Addp_m_AhAdp, [size_kernel(1), size_kernel(2)]) + ...
                    Ah_k( reshape(z(:,:,k,:), [size_z(1),size_z(2),size_z(4)]) );  
            else
                error('Choose an appropriate majorizer type.');
            end
        end
        d = ProxFilterTightFrame( Adnu, size_kernel );
        disp('Restarted: filter update!');
    end
  
        
    %% UPDATE: All sparse codes, { z_{l,k} : l=1,...,L, k=1,...,K }
    
    %Proximal mapping with no majorization
    A_dkxl = A_kl(d);
    z_old = z;
    z = ProxSparseL0( A_dkxl, sqrt(2*alpha) );
    
    
    %% UPDATE: Momentum coeff.  
    tau_old = tau;
    tau = ( 1 + sqrt(1 + 4*tau^2) ) / 2;
    
    %timing
    t_update = toc;

    
    %% EVALUATION    
    %Debug process
    [obj_val, ~, ~] = objective(z, A_dkxl);
    d_relErr = norm( d(:)-d_old(:) ) / norm( d(:) );
    z_relErr = norm( z(:)-z_old(:) ) / norm( z(:) );
    dnorm = norm(d(:),2)^2;     %sanity check
    
fprintf('Iter %d, Obj %3.3g, Filt. Norm %2.2g, D Diff %5.5g, Z Diff %5.5g\n', ...
            i, obj_val, dnorm, d_relErr, z_relErr);
    
    %Record current iteration
    iterations.obj_vals(i + 1) = obj_val;
    iterations.tim_vals(i + 1) = iterations.tim_vals(i) + t_update;
    if mod(i,500) == 0  %save filters every 500 iteration
       iterations.it_vals = cat(4, iterations.it_vals, d); 
    end
    
    %Display intermediate results 
%     if verbose == true
%         if mod(i,50) == 1
%             display_func(iterate_fig, filter_fig, d, A_dkxl, z, x, psf_radius(1), i);
%         end
%     end
%     
    %Termination
    if (d_relErr < tol) && (z_relErr < tol) && (i > 1)
        disp('relErr reached'); 
        break;
    end
    
end

%Final estimate    
d_res = d;
z_res = z;
A_dkxl_res = A_dkxl;

return;





%%%%%%%%%%%%%%%%%%%% Def: Padding Operators %%%%%%%%%%%%%%%%%%%%

function [M, bpad] = PadFunc(b, psf_radius)
    
M = padarray(ones(size(b)), [psf_radius(1), psf_radius(2), 0], 0, 'both');    %mask
%%%circular padding
bpad = padarray(b, [psf_radius, psf_radius, 0], 'circular', 'both');
%%%reflective padding
% bpad = padarray(b, [psf_radius(1), psf_radius(2), 0], 'symmetric', 'both');     
%%%%zero padding
% bpad = padarray(b, [psf_radius, psf_radius, 0], 0, 'both');              
    
return;




%%%%%%%%%%%%%%%%%%%% Def: System Operators %%%%%%%%%%%%%%%%%%%%

function Au = A_for_dk( xpad, u, size_x )

Au = zeros(size_x);
for l = 1 : size_x(3)
    %!!!NOTE: compensate rotating filter in conv2()
    Au(:,:,l) = conv2( xpad(:,:,l), rot90(u,2), 'valid' );
end

return;


function Ahu = Ah_for_dk( xpad, u, size_x, size_kernel )

Ahu = zeros(size_kernel(1), size_kernel(2)); 
for l = 1: size_x(3)
    Ahu = Ahu + conv2( xpad(:,:,l), rot90(u(:,:,l),2), 'valid');    
end

return;


function x_filt = A_for_dk_xl( xpad, d, size_z )

x_filt = zeros(size_z);
for l = 1 : size_z(4)
    for k = 1 : size_z(3)
        %!!!NOTE: compensate rotating filter in conv2()
        x_filt(:,:,k,l) = conv2( xpad(:,:,l), rot90(d(:,:,k),2), 'valid' );
    end
end

return;




%%%%%%%%%%%%%%%%%%%% Design: Majorization Matrices %%%%%%%%%%%%%%%%%%%%

function M =  majorMat_Ak_diag( xpad, size_x, size_kernel )
%Scaled identity majorization matrix in 
%Lem. 5.2 of DOI: 10.1109/TIP.2019.2937734

AtA_symb = zeros(size_kernel(1), size_kernel(2));
for l = 1 : size_x(3)
    P1x = xpad( 1 : 1+size_x(1)-1, 1 : 1+size_x(2)-1, l );
    for r2 = 1 : size_kernel(2)
        for r1 = 1 : size_kernel(1)
            Prx = xpad( r1 : r1+size_x(1)-1, r2 : r2+size_x(2)-1, l );
            AtA_symb(r1, r2) = AtA_symb(r1, r2) + Prx(:)' * P1x(:);
        end
    end
end

M = ( abs(AtA_symb(:))' * ones(size_kernel(1)*size_kernel(2),1) ) .* ...
    ones(size_kernel(1), size_kernel(1));

return;


function M = majorMat_Ak_full( xpad, size_x, size_kernel, L, psf_radius, kern_xgrid, kern_ygrid )  
%Exact Hessian matrix in
%Prop. 5.1 of DOI: 10.1109/TIP.2019.2937734

%!!!NOTE: x-grid and y-grid are horizontal and vertical direction, resp.
%E.g. in matlab, X( y-grid indices, x-grid indices )
kern_xgrid_vec = kern_xgrid(:);
kern_ygrid_vec = kern_ygrid(:);

M = zeros( size_kernel(1)*size_kernel(2), size_kernel(1)*size_kernel(2) );

for k1 = 1 : size_kernel(1)*size_kernel(2)
    for k2 = 1 : k1
        
        k1x_coord = kern_xgrid_vec(k1);
        k1y_coord = kern_ygrid_vec(k1);
        
        k2x_coord = kern_xgrid_vec(k2);
        k2y_coord = kern_ygrid_vec(k2);
        
        for l = 1 : L               
            xpad_k1 = xpad( 1 + psf_radius(1) + k1y_coord : size_x(1) + psf_radius(1) + k1y_coord, ...
                1 + psf_radius(2) + k1x_coord : size_x(2) + psf_radius(2) + k1x_coord, l );

            xpad_k2 = xpad( 1 + psf_radius(1) + k2y_coord : size_x(1) + psf_radius(1) + k2y_coord, ...
                1 + psf_radius(2) + k2x_coord : size_x(2) + psf_radius(2) + k2x_coord, l );

            if k1 == k2
                M(k1, k2) = M(k1, k2) + (xpad_k1(:)'*xpad_k2(:))/2;
            else
                M(k1, k2) = M(k1, k2) + xpad_k1(:)'*xpad_k2(:);
            end            
        end
    end
end

M = M + M';

return;




%%%%%%%%%%%%%%%%%%%% Def: Proximal Operators %%%%%%%%%%%%%%%%%%%%

function d = ProxFilterTightFrame( Adnu, size_kernel )
%Solve orthogonal Procrustes problem

kernVec_size = size_kernel(1)*size_kernel(2);

AdNu = reshape(Adnu, [kernVec_size, size_kernel(3)]);
[U, ~, V] = svd(AdNu, 'econ');

D = sqrt(kernVec_size)^(-1) * U * V';

d = reshape( D, size_kernel );

return;

    

    

%%%%%%%%%%%%%%%%%%%% MISC %%%%%%%%%%%%%%%%%%%%

function [f_val, f_d, sparsity] = objectiveFunction( z, A_dkxl, alpha )

    %Dataterm
    f_d = 1/2 * norm(A_dkxl(:) - z(:), 2)^2;
    %Regularizer
    f_z = nnz(z);

    %Function val
    f_val = f_d + alpha*f_z;
    
    %Sparsity
    sparsity = 100*f_z/numel(z);
    
return;


function [] = display_func(iterate_fig, filter_fig, d, xfilt_d, z, x, psf_radius, iter)

    figure(iterate_fig);
    subplot(4,6,1), imshow(x(:,:,2),[]); axis image; colormap gray; title(sprintf('Local iterate %d',iter));
    
    subplot(4,6,2), imshow(xfilt_d(:,:,1,2),[]); axis image; colormap gray; title('Filt. img');
    subplot(4,6,3), imshow(xfilt_d(:,:,6,2),[]); axis image; colormap gray;
    subplot(4,6,4), imshow(xfilt_d(:,:,11,2),[]); axis image; colormap gray;
    subplot(4,6,5), imshow(xfilt_d(:,:,16,2),[]); axis image; colormap gray;
    subplot(4,6,6), imshow(xfilt_d(:,:,21,2),[]); axis image; colormap gray;
        
    subplot(4,6,8), imshow(z(:,:,1,2),[]); axis image; colormap gray; title('Spar. code'); 
    subplot(4,6,9), imshow(z(:,:,6,2),[]); axis image; colormap gray;
    subplot(4,6,10), imshow(z(:,:,11,2),[]); axis image; colormap gray;
    subplot(4,6,11), imshow(z(:,:,16,2),[]); axis image; colormap gray;
    subplot(4,6,12), imshow(z(:,:,21,2),[]); axis image; colormap gray;
    
    subplot(4,6,13), imshow(x(:,:,4),[]); axis image; colormap gray; title(sprintf('Local iterate %d',iter));
    
    subplot(4,6,14), imshow(xfilt_d(:,:,1,4),[]); axis image; colormap gray; title('Filt. img');
    subplot(4,6,15), imshow(xfilt_d(:,:,6,4),[]); axis image; colormap gray;
    subplot(4,6,16), imshow(xfilt_d(:,:,11,4),[]); axis image; colormap gray;
    subplot(4,6,17), imshow(xfilt_d(:,:,16,4),[]); axis image; colormap gray;
    subplot(4,6,18), imshow(xfilt_d(:,:,21,4),[]); axis image; colormap gray;

    subplot(4,6,20), imshow(z(:,:,1,4),[]); axis image; colormap gray; title('Spar. code'); 
    subplot(4,6,21), imshow(z(:,:,6,4),[]); axis image; colormap gray;
    subplot(4,6,22), imshow(z(:,:,11,4),[]); axis image; colormap gray;
    subplot(4,6,23), imshow(z(:,:,16,4),[]); axis image; colormap gray;
    subplot(4,6,24), imshow(z(:,:,21,4),[]); axis image; colormap gray;
    drawnow;

    figure(filter_fig);
    sqr_k = ceil(sqrt(size(d,3)));
    pd = 1;
    d_disp = zeros( sqr_k * [psf_radius*2+1 + pd, psf_radius*2+1 + pd] + [pd, pd]);
    for j = 0:size(d,3) - 1
        d_curr = d(:,:,j+1);
        d_disp( floor(j/sqr_k) * (size(d_curr,1) + pd) + pd + (1:size(d_curr,1)) , mod(j,sqr_k) * (size(d_curr,2) + pd) + pd + (1:size(d_curr,2)) ) = d_curr;
    end
    imagesc(d_disp); colormap gray; axis image; colorbar; 
    title(sprintf('Filter iterate %d',iter));
    drawnow;
        
return;
