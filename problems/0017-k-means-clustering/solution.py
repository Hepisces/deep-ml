import numpy as np
def distance(a,b):
	return np.sqrt(np.sum((a-b)**2,axis=1))


def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
	points=np.array(points)
	centroids = np.array(initial_centroids)

	for iteration in range(max_iterations):
		distances=np.array([distance(points,centroid) for centroid in centroids])
		assignments = np.argmin(distances,axis=0)

		new_centroids= np.array([points[assignments==i].mean(axis=0) if len(points[assignments == i])>0 else centorids[i] for i in range(k)])

		if np.all(centroids == new_centroids):
			break
		centroids = new_centroids
		centroids = np.round(centroids,4)
	return [tuple(centroid) for centroid  in centroids]