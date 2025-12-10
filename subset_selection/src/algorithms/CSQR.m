function [p] = CSQR(Q,p,e)
  %U is an orthonormal matrix
  %p is a set of pre-selected indices
  %e is the number of oversampling indices

  k = max(size(p)); %Determine number of selections
  if e <= 0
    p=p;
  elseif k < e
  while max(size(p)) < k + e
  [~,S,V] = svd(Q(p,:),'econ'); %Find the right singular vectors of the selected indices of Q
  Qcomplement = Q;
  Qcomplement(p,:) = []; %Remove the selected rows from Q
  CminusE = diag(1./diag(sqrt(diag(ones(max(size(S))))-S.^2)));
  [~,~,extraIndices] = qr((Qcomplement*V*CminusE)','vector');
  p = [p extraIndices];
  end
  p = p(:,1:k+e)';

  else
  [~,S,V] = svd(Q(p,:),'econ'); %Find the right singular vectors of the selected indices of Q
  VminusE = V(:,k-e+1:k); %Truncate to the trailing e right singular vectors
  SminusE = S(k-e+1:k,k-e+1:k);
  CminusE = diag(1./diag(sqrt(diag(diag(ones(e)))-S(k-e+1:k,k-e+1:k).^2)));
  Qcomplement = Q;
  Qcomplement(p,:) = []; %Remove the selected rows from U
  [~,~,extraIndices] = qr((Qcomplement*VminusE*CminusE)','vector');
  p = [p extraIndices];
  p = p(:,1:k+e)';
  end


