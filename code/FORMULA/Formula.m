function [ U, V, info ] = Formula( X, Y, L, lambda1, lambda2, lambda3, k )
%FORMULAJOINT Summary of this function goes here
%   Detailed explanation goes here

% This function is the main function of algorithm Formula.
% Input:
% -- X: NxD design matrix
% -- Y: response variable
% -- L: task similarity matrix
% -- lambda1, lambda2, lambda3: parameters for regularizations
% -- k: number of hidden tasks
% --    lambda1 : sparse for V
% --    lambda2 : sparse for U
% --    lambda3 : model neighborhood reconstruction. 
%
% Output:
% -- U: DxK matrix -- basis for models
% -- V: KxN matrix -- task assignment matrix

% standardization X
X = zscore(X);
[n, d] = size(X);

% initialization U and V randomly. Other initialization methods are also possible.
rng(0);
U = rand(d,k);
V = rand(k,n);

UV_vect0 = [U(:); V(:)];


% function value/gradient of the smooth part
smoothF    = @(UV_vector) smooth_part( UV_vector, X, Y, L, lambda3);
% non-negativen l1 norm proximal operator.
non_smooth = prox_nnl1_Vonly(lambda1, lambda2,  size(U, 1) * size(U, 2) + 1 );

sparsa_options = pnopt_optimset(...
    'display'   , 0    ,...
    'debug'     , 0    ,...
    'maxIter'   , 1000  ,...
    'ftol'      , 1e-5 ,...
    'optim_tol' , 1e-5 ,...
    'xtol'      , 1e-5 ...
    );
[UV_vect, ~,info] = pnopt_sparsa( smoothF, non_smooth, UV_vect0, sparsa_options );


% U=> d * k
U = reshape (UV_vect(1:d*k), [d , k]);
% V=> k * n (remaining)
V = reshape (UV_vect(d*k + 1:end), [k , n]);

end

function [f, g] = smooth_part(UV_vect, X, Y, L, lambda3)

[n, d] = size(X);
k = length(UV_vect)/(n + d);
% U=> d * k
U = reshape (UV_vect(1:d*k), [d , k]);
% V=> k * n (remaining)
V = reshape (UV_vect(d*k + 1:end), [k , n]);


% gradient w.r.t U
A = V * (eye(n) - L);
gradientU = lambda3 * U * (A * A');
for i = 1:n
    xi = X(i,:)';
    vi = V(:,i);
    tmp = - Y(i)* (xi * vi') + xi * (xi' * U * vi) * vi';
    gradientU = gradientU + tmp;
end

% gradient w.r.t V
X_tilde = X*U;
B = eye(n) - L;
P = -ones(k,1) * Y' .* X_tilde';
Q = (X_tilde.* repmat(sum(X_tilde .* V', 2), [1, k]))';

gradientV = P + Q + lambda3 * (U' * U) * ((V * B) * B');

% function value
f = 0.5 * sum((Y' - (sum(V.*X_tilde'))).^2) ...
    + lambda3/2 * norm(U*A, 'fro')^2;
% joint gradient
g = [gradientU(:); gradientV(:)];

end


function op = prox_nnl1_Vonly( q , r,  V_startIdx)

if nargin == 0,
    q = 1;
elseif ~isnumeric( q ) || ~isreal( q ) ||  any( q < 0 ) || all(q==0) %|| numel( q ) ~= 1
    error( 'Argument must be positive.' );
end

op = tfocs_prox( @f, @prox_f , 'vector' ); % Allow vector stepsizes

    function v = f(x)
        v = norm( q*x(V_startIdx:end), 1 ) + norm( r*x(1:V_startIdx-1), 1 );
    end

    function x = prox_f(x,t)
        tq = t .* q; 
        tr = t .* r; 
        
        % project U
        x(1:V_startIdx-1) = max(x(1:V_startIdx-1) - tq, 0);
        % project V
        x(V_startIdx:end) = max(x(V_startIdx:end) - tr, 0);
        
    end


end


