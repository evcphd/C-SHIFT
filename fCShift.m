function [ Cov_CSHIFT, Corr_CSHIFT] = fCShift(Cov_obs, geneNo)
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%
% DESCRIPTION: This function normalizes observed (noisy) covariance matrix
% using C-SHIFT method 
%
% INPUT:
% Cov_obs - observed covariance matrix of size geneNo x geneNo
% geneNo - number of genes (note it is denoted as M in the paper)
%
% OUTPUT:
% Cov_CSHIFT - normalized covariance matrix of size geneNo x geneNo
% Corr_CSHIFT  - normalized correlation matrix of size geneNo x geneNo
%
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% More information about build-in function used: 
% https://www.mathworks.com/help/optim/ug/fminunc.html
% https://www.mathworks.com/help/optim/ug/fmincon.html#busog7r-fun
% https://www.mathworks.com/help/optim/ug/fmincon.html#busog7r-options
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% if Cov_obs is not a full rank then we add additional diagonal 
% perturbation matrix for better inversion
if rank(Cov_obs)==geneNo
    C_tilde = Cov_obs;
else
    F = diag(rand(1,geneNo));
    C_tilde = Cov_obs+F;
end

% initial alpha
alpha = zeros(geneNo,1); 

% set the current algorithm to 'trust-region' (note that defualt option 
% is 'quasi-newton')
options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective');
% define the objective function, its gradient, and its Hessian
f = @ObjFunct;

    function [obj_funct,grad,hess] = ObjFunct(alpha)
        
        % auxiliary variables for objective function 
        D = repmat(alpha,1,geneNo);     % is alpha*1^(T) in the paper
        A_alpha = C_tilde+D+D';         % is A_alpha in the paper
        A_hlp = A_alpha\ones(geneNo,1); % is (A_alpha)^(-1)*1 in the paper
        V = 1/(ones(1,geneNo)*A_hlp);   % is V(alpha) in the paper
        % objective function
        obj_funct = norm(A_alpha-V*ones(geneNo,geneNo),'fro')^2;
        
        % auxiliary variables for the gradient
        a = sum(alpha);         % is a in the paper
        c = sum(sum(C_tilde));  % is c in the paper
        hlp1 = geneNo^2*V^2-c*V-2*geneNo*a*V;
        % gradient of the objective function
        grad = 4*(geneNo*alpha + hlp1*A_hlp + sum(C_tilde,2) + (a-geneNo*V)*ones(geneNo,1));
        
        % auxiliary variables for the Hessian
        hlp2 = A_hlp*ones(1,geneNo); 
        H1 = hlp2+hlp2';
        H2 = A_hlp*A_hlp';
        term2 = (3*geneNo^2*V-c-2*geneNo*a)*V*H2;
        term3 = (geneNo^2*V-c-2*geneNo*a)*inv(A_alpha); % scalar*inv(M_hlp), do not use scalar/M_hlp
        term1 = 2*geneNo*V*H1;
        % Hessian
        hess = 4*(diag(geneNo*ones(1,geneNo))+ones(geneNo,geneNo)-term1+term2-term3);  
        
     
        if norm(hess-hess','fro')<0.01
            disp('hess is symmetric')
            disp('frobnorm(Hess-hess)')
            disp(norm(hess-hess','fro'))
        else
            disp('hess is not symmetric')
            disp('frobnorm(Hess-hess)')
            disp(norm(hess-hess','fro'))
        end
        
    end

% solution to the min problem
alpha_sol = fminunc(f,alpha,options);

% calculate the normalized covariance matrix Cov_CSHIFT that depends on 
% the newly found alpha_sol
D = repmat(alpha_sol,1,geneNo);
M = C_tilde+D+D';
A_hlp = M\ones(geneNo,1);
V = 1/(ones(1,geneNo)*A_hlp);
C_alpha_sol = M-V*ones(geneNo,geneNo);

% normalized covariance matrix (adjusted)
Cov_CSHIFT = C_alpha_sol-F; 

% normalized correlation matrix 
Corr_CSHIFT = zeros(geneNo,geneNo);
for m = 1:geneNo
    for n = m:geneNo
        Corr_CSHIFT(m,n) = (Cov_CSHIFT(m,n))/(sqrt((Cov_CSHIFT(n,n))*(Cov_CSHIFT(m,m))));
        Corr_CSHIFT(n,m) = Corr_CSHIFT(m,n);
    end
end
end


