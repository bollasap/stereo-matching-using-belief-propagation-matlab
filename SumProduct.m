dispLevels = 16;
iterations = 80;
lambda = 5;
threshold = dispLevels-1; %don't use threshold

% Set the disparity values
d = 0:dispLevels-1;

% Read the stereo image and convert it to grayscale
left = rgb2gray(imread('Left.png'));
right = rgb2gray(imread('Right.png'));

% Apply a Gaussian filter
left = imgaussfilt(left,0.6,'FilterSize',5);
right = imgaussfilt(right,0.6,'FilterSize',5);

% Get the image size
[height,width] = size(left);

% Compute data term
dataTerm = zeros(height,width,dispLevels);
for i = 1:dispLevels
	right_d = [zeros(height,d(i)),right(:,1:end-d(i))];
	dataTerm(:,:,i) = exp(-abs(double(left)-double(right_d)));
end

% Compute smoothness term
smoothnessTerm = exp(-lambda*min(abs(d-d'),threshold));

% Initialize messages
msgUp = ones(height,width,dispLevels);
msgDown = ones(height,width,dispLevels);
msgRight = ones(height,width,dispLevels);
msgLeft = ones(height,width,dispLevels);

figure

% Start iterations
for i = 1:iterations
	% Auxiliary tables that help us create the messages
	U = dataTerm .* msgDown .* msgRight .* msgLeft;
	D = dataTerm .* msgUp .* msgRight .* msgLeft;
	R = dataTerm .* msgUp .* msgDown .* msgLeft;
	L = dataTerm .* msgUp .* msgDown .* msgRight;
	
	% For each pixel
	for y = 2:height-1
		for x = 2:width-1
			% Send message up
			msg = reshape(U(y,x,:),[dispLevels,1]);
			msg = sum(msg.*smoothnessTerm);
			msg = msg/sum(msg);
			msgDown(y-1,x,:) = msg;
			
			% Send message down
			msg = reshape(D(y,x,:),[dispLevels,1]);
			msg = sum(msg.*smoothnessTerm);
			msg = msg/sum(msg);
			msgUp(y+1,x,:) = msg;
			
			% Send message right
			msg = reshape(R(y,x,:),[dispLevels,1]);
			msg = sum(msg.*smoothnessTerm);
			msg = msg/sum(msg);
			msgLeft(y,x+1,:) = msg;
			
			% Send message left
			msg = reshape(L(y,x,:),[dispLevels,1]);
			msg = sum(msg.*smoothnessTerm);
			msg = msg/sum(msg);
			msgRight(y,x-1,:) = msg;
		end
	end
	
	% Compute belief
	belief = dataTerm .* msgUp .* msgDown .* msgRight .* msgLeft;
	
	% Update disparity map
	[Y,I] = max(belief,[],3);
	dispMap = d(I);
	
	% Update disparity image
	scaleFactor = 256/dispLevels;
	dispImage = uint8(dispMap*scaleFactor);
	
	% Show disparity image
	imshow(dispImage)
	
	% Show current iteration
	fprintf('iteration %d/%d\n',i,iterations)
end

% Save disparity image
imwrite(dispImage,'Disparity.png')