function [p] = CSDEIM(U,p,e)
  %U is an orthonormal matrix
  %p is a set of pre-selected indices
  %e is the number of oversampling indices

  k = max(size(p)); %Determine number of selections
  if e <= 0
    p=p;
  elseif k < e
  while max(size(p)) < k + e
  [~,S,V] = svd(U(p,:),'econ'); %Find the right singular vectors of the selected indices of U
  Ucomplement = U;
  Ucomplement(p,:) = []; %Remove the selected rows from U
  CminusE = diag(1./diag(sqrt(diag(ones(max(size(S))))-S.^2)));
  %[extraIndices] = Deim(Ucomplement*V);
  [extraIndices] = Deim(Ucomplement*V*CminusE);
  p = [p; extraIndices];
  end
  p = p(1:k+e,:);

  else
  [~,S,V] = svd(U(p,:),'econ'); %Find the right singular vectors of the selected indices of U
  VminusE = V(:,k-e+1:k); %Truncate to the trailing e right singular vectors
  SminusE = S(k-e+1:k,k-e+1:k);
  CminusE = diag(1./diag(sqrt(diag(diag(ones(e)))-S(k-e+1:k,k-e+1:k).^2)));
  Ucomplement = U;
  Ucomplement(p,:) = []; %Remove the selected rows from U
  %[extraIndices] = Deim(Ucomplement*VminusE);
  [extraIndices] = Deim(Ucomplement*VminusE*CminusE);
  p = [p; extraIndices];
  end


