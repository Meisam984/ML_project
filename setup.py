from setuptools import find_packages, setup
from typing import List

HYPHEN_DOT_E = '-e .'


def fetch_requirements(file_path: str) -> List[str]:
	with open(file=file_path, mode='r') as req_file:
		req_list = req_file.readlines()
		req_list = [req.replace('\n', '') for req in req_list]

		if HYPHEN_DOT_E in req_list:
			req_list.remove(HYPHEN_DOT_E)

	return req_list


setup(
	name='ML-Project',
	version='0.0.1',
	author='Meisam Rezaei',
	author_email='meysam.or.us@gmail.com',
	url='https://github.com/Meisam984/ML_project.git',
	packages=find_packages(),
	install_requires=fetch_requirements('requirements.txt')
)
