from NBodySimulator import NBodySimulator 

sim = NBodySimulator(n=3, dim=2, masses=[1, 2, 4])
pos0, vel0 = sim.random_initial_conditions()
tracks = sim.simulate(pos0, vel0, t_end=2.0, dt=0.001, visualize=True, energy_check=True)
NBodySimulator.plot_tracks(tracks)
NBodySimulator.plot_energy(tracks)
sim.save(tracks, "run_001.npz")