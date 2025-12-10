function [p] = QDEIMandRestartedDEIMv2(U,extension_method,tol,beta)
%
%    This function selects the extended DEIM points p
%    for the columns of U. U is assumed to be full rank.
%
%    Input:
%               U - an nxm full rank matrix
%               extension_method - a string indicating the extended DEIM method
%                   of choice. Options are:
%                       'no_mem'--> Extended DEIM without accounting for
%                           the indices selected in the standard DEIM
%                           implementation
%                       'mem_l1'--> Extended DEIM using the l-1 norm in
%                           comparing remaining rows/columns with those
%                           selected via standard DEIM
%                       'mem_coh'--> Extended DEIM using coherence in
%                           comparing remaining rows/columns with those
%                           selected via standard DEIM
%                       'mem_DTW'--> Extended DEIM using dynamic time warping (DTW) in
%                           comparing remaining rows/columns with those
%                           selected via standard DEIM
%               tol - the extended DEIM tolerance for determining whether
%                           a new index is added (corresponding to the method chosen)
%               beta - the maximum number of row indices to be selected through
%                           extended DEIM; must be >= rank of U and < n.
%
% Output:
%               p - the row indices selected through extended DEIM


[n,m] = size(U);

if nargin < 3
    tol = 1e-10; % This can be adjusted depending on how large the residual is allowed to be
end

if nargin < 4
    beta = 2*m; % This can be adjusted depending on how many rows are to be selected
end

if beta > n
    disp('Cannot select the number of indices specified by beta because beta exceeds the number of rows in U. Selecting the maximum possible number of rows.')
end


m = min([m,beta]);
% Perform QR
p = zeros(m,1);
[~, ~, p] = qr(U','vector') ;
p = p(1,1:m)' ;

stop_flag = 0;

while length(p) < beta && stop_flag == 0
% while length(p) < beta

    % Identify indices not yet selected
    indices = 1:n;

    kept = indices(~ismember(indices,p));

    % Carry out extension of DEIM
    if ~isempty(kept) % Added condition 4-4-17

        U_hat = U(kept,:);

        % Form r_2 according to extension of interest:
        if strcmpi(extension_method,'no_mem')
            % No "memory"
            r2 = ones(length(kept),1);

            % Form the "memory" residual vector, r_2, using the l-1 norm
        elseif strcmpi(extension_method,'mem_l1')
            remain_resid = zeros(length(kept),length(p));
            for rem = 1:length(p)
                remain_resid(:,rem) = sum(abs(ones(length(kept),1)*U(p(rem),:) - U_hat),2);
            end
            r2 = min(remain_resid,[],2);
            r2 = r2/max(r2);

        elseif strcmpi(extension_method,'mem_coh')
            % compute the coherence between U_hat and U, and flip scale so that those
            % with lower coherence are given more weight
            r2 = 1 - max(abs(normr(U_hat)*(normr(U(p,:))')),[],2);

        elseif strcmpi(extension_method,'mem_DTW')
            %%%%%%%%%%%%%% Only use with Matlab 2016a or a later version %%%%%%%%%%%%%
            remain_resid = zeros(length(kept),length(p));
            for rem = 1:length(p)
                for rem_row = 1:length(kept)
                    % Compute DTW distance
                    remain_resid(rem_row,rem) = dtw(U(p(rem),:),U_hat(rem_row,:)); % Added 3-21-17
                end
            end
            r2 = min(remain_resid,[],2);
            r2 = r2/max(r2);

        else
            display('Not a valid DEIM extension.')
            return
        end



        % Restart DEIM incorporating information from r_2 and skipping columns
        % that do not contain enough new information
        i = 1;
        rho = 0;
        while rho <= tol && i <= m
            [rho,p_hat] = max(abs(U_hat(:,i).*r2));
            i  = i + 1;
        end


        if rho <= tol && i == m+1
% %             p_hat_full = []; % nothing to be added to selected indices
% %             stop_flag = 1; % set flag to stop looking for more indices
            disp('residuals too small to add more indices')
            break
        else

            indx = i-1;

            while i <= m && (length(indx) < (beta-length(p)))
                u = U_hat(:,i);

                r1 = u - U_hat(:,indx)*(U_hat(p_hat,indx)\u(p_hat));

                r_hat = r1.*r2;
                [val,p_i] = max(abs(r_hat));

                if val > tol
                    p_hat = [p_hat; p_i];
                    indx = [indx, i];
                end
                i = i + 1;
            end

            p_hat_full = kept(p_hat);
        end

        p = [p; p_hat_full(:)];
    else
        stop_flag = 1;
    end

end
