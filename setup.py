from setuptools import setup

setup(
    name='XtrUtils',
    version='1.0',
    author='Aaron Gerston (X-trodes LTD)',
    author_email='aarong@xtrodes.com',
    packages=['XtrUtils'],
    include_package_data=True,
    license='GNU GPLv3',
    long_description=open('README.md').read(),
    # url="",
    install_requires=open('requirements.txt').read()
)
