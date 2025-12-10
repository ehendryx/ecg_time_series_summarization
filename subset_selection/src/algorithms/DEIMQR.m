function [p] = DEIMQR(U,k)
##  p = [];
##  l = [];
  [m,n] = size(U);
  V = U(:,1);
  khat = k - n;
  e = min([k,n]);
  p = zeros(k,1);

  % determine first e indices using deim
  [rho, p(1,1)] = max(abs(U(:,1)));
  for j = 2:e
    u = U(:,j);
    r = u - U(:,1:j-1)*(U(p(1:j-1),1:j-1)\u(p(1:j-1)));
    V = [V r]; % Residual Vector storage
    [rho,p(j,1)] = max(abs(r)); % selects next DEIM index
  end

  % remainder of algorithm determines last khat indices > 0
  if khat > 0
##  [~, ~, pprime] = qr(V','vector');
##  pprime = pprime(1:e);
##  p(e+1:2*e,1) = pprime';
##  V(p(1:2*e,1),:) = zeros(2*e,e);
  for i = 1:(ceil(abs((khat-e)/e))-1)
    [~, ~, pprime] = qr(V','vector');
    pprime = pprime(1:e);
    p((i)*e+1:(i+1)*e,1) = pprime';
    V(p(1:(i+1)*e,1),:) = zeros((i+1)*e,e);
  end
  [~, ~, pprime] = qr(V','vector');
  pprime = pprime(1:k-ceil(abs((khat-e)/e))*e);
  p(ceil(abs((khat-e)/e))*e+1:k,1) = pprime';
  end
