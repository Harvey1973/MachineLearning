function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));
nm = num_movies;
nu = num_users;
sum_j= 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%J = ;

for i = 1:num_movies
    idx_user = find(R(i,:)==1); %list of users rated movie i
    Theta_temp=Theta(idx_user,:);
    Y_temp = Y(i,idx_user);
    X_grad(i,:) = (X(i,:)*transpose(Theta_temp)-Y_temp)*Theta_temp + lambda*X(i,:);
endfor
sum_j = 0;
sum_x = 0;

for j = 1 :num_users 
    idx_movie = find(R(:,j)==1) ; %list of movies that user j rated
    
    x_temp  = X(reshape(idx_movie,1,size(idx_movie)),:);
    y_temp = Y(idx_movie,j);
    Theta_grad(j,:) = transpose((x_temp*transpose(Theta(j,:))-y_temp))*x_temp + lambda*Theta(j,:);
endfor

for i = 1:num_users
    for k =1: num_features  
        sum_j =sum_j + Theta(i,k)^2;
    endfor
endfor
for i = 1:num_movies
    for k =1: num_features  
        sum_x =sum_x + X(i,k)^2;
    endfor
endfor
%+ lambda*sum(sum(transpose(Theta(:,[2:size(Theta,2)]))*Theta(:,[2:size(Theta,2)])))/2+lambda*sum(sum(transpose(X(:,[2:size(X,2)]))*X(:,[2:size(X,2)])))/2 
J = sum(sum(R.*(X*transpose(Theta)-Y).^2))/2 +lambda*sum_j/2 + lambda*sum_x/2;








% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
