from setuptools import setup, find_packages

package_name = 'dss_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'nats-py', 'PyYAML', 'protobuf', 'opencv-python', 'numpy'],
    zip_safe=True,
    maintainer='smjeon',
    maintainer_email='smjeon@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dss_bridge_node = dss_bridge.dss_bridge_node:main',
            'dss_car_control_node = dss_bridge.dss_car_control_node:main',
        ],
    },
)
