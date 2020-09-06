
function mu = softmaxx(eta)

    c = 3;

    tmp = exp(c*eta);
    denom = sum(tmp, 2);
    mu = bsxfun(@rdivide, tmp, denom);

end
