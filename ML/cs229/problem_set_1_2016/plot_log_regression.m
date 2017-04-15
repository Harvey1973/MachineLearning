
X = load('logistic_x.txt');
Y = load('logistic_y.txt');

X = [ones(size(X,1),1) X];
[theta,l1] = log_regression(X,Y,20);

m = size(X,1);
figure; hold on;

%plot the examples 
plot(X(Y < 0, 2),X(Y < 0,3),'rx','linewidth',2);
plot(X(Y > 0, 2),X(Y > 0,3),'go','linewidth',2);

% plot the decision boundary , note that the response is given by y =
% theta(1) + theta(2)*x1+theta(3)*x2  the boundry is given by setting y = 0
% so x2 = (-theta(1))/theta(3) - (theta(2)/theta(3))*x1

x1 = min(X(:,2)):0.01:max(X(:,2));
x2 = (-theta(1))/theta(3) - (theta(2)/theta(3))*x1;

plot(x1,x2,'linewidth',2);
xlabel('x1');
ylabel('x2');

