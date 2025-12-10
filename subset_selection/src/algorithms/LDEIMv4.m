function [p] = LDEIMv4(U,k)
##  p = [];
##  l = [];
  [m,n] = size(U);
  V = U(:,1);
  khat = k - n;
  e = min([k,n]);
  p = zeros(k,1);
  l = zeros(m,1);

  % determine first e indices using deim
  [rho, p(1,1)] = max(abs(U(:,1)));
  for j = 2:e
    u = U(:,j);
    r = u - U(:,1:j-1)*(U(p(1:j-1),1:j-1)\u(p(1:j-1)));
    V = [V r]; % Residual Vector storage
    [rho,p(j,1)] = max(abs(r)); % selects next DEIM index
  end

  % remainder of algorithm determines last khat indices > 0
  if khat >= 0
  for i = 1:m
    l(i,1) = norm(V(i,:));
  end

  [o,t] = sort(l,'descend');
  t(ismember(t,p)) = [];
  pprime = t(1:khat,1);
  pprime(pprime==0) = [];
  p(k-khat+1:k,1) = pprime;
  end
