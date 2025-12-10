function Y = normr(X)
  nrm = sqrt(sum(X.^2, 2));
  Y = X;
  idx = nrm > 0;
  Y(idx, :) = X(idx, :) ./ nrm(idx);
end

