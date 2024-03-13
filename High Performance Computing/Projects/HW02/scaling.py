import matplotlib.pyplot as plt

# Read the timings from the file
with open('timings.txt', 'r') as f:
    lines = f.readlines()
    ns = [int(line.split()[0]) for line in lines]
    timings = [float(line.split()[1]) for line in lines]

plt.figure(figsize=(10, 6))
plt.plot(ns, timings, '-o', label="Scan Time")
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('n')
plt.ylabel('Time (milliseconds)')
plt.title('Scaling Analysis of Scan Algorithm')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig('task1.pdf')